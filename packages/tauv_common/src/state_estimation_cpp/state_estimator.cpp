#include <boost/circular_buffer.hpp>
#include <boost/heap/priority_queue.hpp>
#include <boost/variant.hpp>
#include <eigen3/Eigen/Dense>
#include <mutex>
#include <nav_msgs/Odometry.h>
#include <numeric>
#include <ros/ros.h>
#include <tauv_msgs/FluidDepth.h>
#include <tauv_msgs/NavigationState.h>
#include <tauv_msgs/TeledyneDvlData.h>
#include <tauv_msgs/XsensImuData.h>
#include <tf/transform_broadcaster.h>
#include <tauv_alarms/alarms.h>
#include <tauv_alarms/alarm_client.h>
#include <tauv_msgs/SetPose.h>
#include "ekf.h"
#include "state_estimator.h"

#define CHECKPOINT_BUFFER_SIZE 400
#define DELAYED_QUEUE_SIZE 100
#define REALTIME_QUEUE_SIZE 100
#define TOPIC_QUEUE_SIZE 100

StateEstimator::StateEstimator(ros::NodeHandle& n, ros::NodeHandle &pn) : n(n), pn(pn), alarm_client(n)
{
  this->load_config();

  this->is_initialized = false;
  this->last_angular_velocity = Eigen::Vector3d::Zero();

  this->ekf = Ekf();
  this->ekf.set_dvl_offset(this->dvl_offset);
  this->ekf.set_process_covariance(this->process_covariance);

  this->previous_angular_acceleration = Eigen::Vector3d::Zero();

  this->checkpoints = boost::circular_buffer<StateEstimator::Checkpoint>(CHECKPOINT_BUFFER_SIZE);
  this->delayed_queue.reserve(DELAYED_QUEUE_SIZE);
  this->realtime_queue.reserve(REALTIME_QUEUE_SIZE);

  this->imu_sub = n.subscribe("vehicle/xsens_imu/data", TOPIC_QUEUE_SIZE, &StateEstimator::handle_imu, this);
  this->dvl_sub = n.subscribe("vehicle/teledyne_dvl/data", TOPIC_QUEUE_SIZE, &StateEstimator::handle_dvl, this);
  this->depth_sub = n.subscribe("vehicle/arduino/depth", TOPIC_QUEUE_SIZE, &StateEstimator::handle_depth, this);

  this->navigation_state_pub = n.advertise<tauv_msgs::NavigationState>("gnc/navigation_state", TOPIC_QUEUE_SIZE);
  this->odom_pub = n.advertise<nav_msgs::Odometry>("gnc/odom", TOPIC_QUEUE_SIZE);

  this->set_pose_srv = n.advertiseService("gnc/state_estimation/set_pose", &StateEstimator::handle_set_pose, this);

  this->timer = n.createTimer(this->dt, &StateEstimator::update, this);

  this->alarm_client.clear(tauv_alarms::AlarmType::STATE_ESTIMATION_NOT_INITIALIZED, "State estimation initialized.");
}

void StateEstimator::load_config()
{
  int frequency;
  this->pn.getParam("frequency", frequency);
  this->dt = ros::Duration(1.0 / double(frequency));

  this->n.getParam("max_delayed_queue_size", this->max_delayed_queue_size);

  double horizon_delay;
  this->pn.getParam("horizon_delay", horizon_delay);
  this->horizon_delay = ros::Duration(horizon_delay);

  std::vector<double> dvl_offset;
  this->pn.getParam("dvl_offset", dvl_offset);
  this->dvl_offset = Eigen::Map<Eigen::Vector3d>(dvl_offset.data());

  std::vector<double> process_covariance;
  this->pn.getParam("process_covariance", process_covariance);
  this->process_covariance = Eigen::Map<Eigen::Matrix<double, 15, 1>>(process_covariance.data());

  std::vector<double> imu_covariance;
  this->pn.getParam("imu_covariance", imu_covariance);
  this->imu_covariance = Eigen::Map<Eigen::Matrix<double, 9, 1>>(imu_covariance.data());

  std::vector<double> dvl_covariance;
  this->pn.getParam("dvl_covariance", dvl_covariance);
  this->dvl_covariance = Eigen::Map<Eigen::Vector3d>(dvl_covariance.data());

  std::vector<double> depth_covariance;
  this->pn.getParam("depth_covariance", depth_covariance);
  assert(depth_covariance.size() == 1);
  this->depth_covariance = depth_covariance.front();

  this->n.getParam("tf_namespace", this->tf_namespace);

  this->pn.getParam("euler_acceleration_filter_constant", this->euler_acceleration_filter_constant);
}

void StateEstimator::update(const ros::TimerEvent& e)
{
  if (!this->is_initialized) return;

  ros::Time current_time = ros::Time::now() - this->horizon_delay;

  ROS_DEBUG("delayed: %ld, realtime: %ld", this->delayed_queue.size(), this->realtime_queue.size());

  if (this->delayed_queue.size() > (unsigned) this->max_delayed_queue_size) {
      this->alarm_client.set(tauv_alarms::AlarmType::STATE_ESTIMATION_DELAYED, "Delayed queue is too large.");
  } else {
      this->alarm_client.clear(tauv_alarms::AlarmType::STATE_ESTIMATION_DELAYED, "Delayed queue is acceptable.");
  }

  if (!this->delayed_queue.empty() && this->checkpoints.full()) {
    boost::shared_ptr<SensorMsg> msg = this->delayed_queue.top();
    this->delayed_queue.pop();

    boost::circular_buffer<Checkpoint>::iterator it = this->checkpoints.begin();
    Checkpoint previous_checkpoint = *it;
    while (it != this->checkpoints.end() && it->msg->stamp <= msg->stamp) {
      previous_checkpoint = *it;
      ++it;
    }
    if (previous_checkpoint.msg->stamp > msg->stamp) {
      ROS_WARN("previous checkpoint stamp later than msg stamp");
    } else {
        this->ekf.set_state(previous_checkpoint.state);
        this->ekf.set_cov(previous_checkpoint.cov);
        this->ekf.set_time(previous_checkpoint.msg->stamp.toSec());

        boost::circular_buffer<Checkpoint>::iterator copy_it = it;
        boost::heap::priority_queue<boost::shared_ptr<SensorMsg>, boost::heap::compare<std::greater_equal<boost::shared_ptr<SensorMsg>>>> msgs;
        msgs.push(msg);
        while (copy_it != this->checkpoints.end()) {
          msgs.push(copy_it->msg);
          ++copy_it;
        }

        this->checkpoints.erase_end(std::distance(it, this->checkpoints.end()));

        while (!this->delayed_queue.empty()) {
          boost::shared_ptr<SensorMsg> msg = this->delayed_queue.top();
          this->delayed_queue.pop();
          msgs.push(msg);
        }

        while (!msgs.empty()) {
          boost::shared_ptr<SensorMsg> msg = msgs.top();
          msgs.pop();

          this->apply_msg(msg);
        }
    }
  }

  while (!this->realtime_queue.empty()) {
    boost::shared_ptr<StateEstimator::SensorMsg> msg = this->realtime_queue.top();
    this->realtime_queue.pop();

    this->apply_msg(msg);
  }


  Eigen::Vector3d position, velocity, acceleration, orientation, angular_velocity;
  this->ekf.get_state_fields(current_time.toSec(), position, velocity, acceleration, orientation, angular_velocity);

  Eigen::Vector3d angular_acceleration = (angular_velocity - this->last_angular_velocity) / (current_time - this->last_evaluation_time).toSec();
  angular_acceleration = this->euler_acceleration_filter_constant * angular_acceleration + (1 - this->euler_acceleration_filter_constant) * (this->previous_angular_acceleration);
  this->previous_angular_acceleration = angular_acceleration;

  this->last_evaluation_time = current_time;
  this->last_angular_velocity = angular_velocity;

  this->publish_navigation_state(current_time, position, velocity, acceleration, orientation, angular_velocity, angular_acceleration);
  this->publish_odom(current_time, position, velocity, orientation, angular_velocity);
  this->publish_tf(current_time, position, orientation);
}

void StateEstimator::apply_msg(boost::shared_ptr<SensorMsg> msg)
{
    switch (msg->type) {
      case SensorMsg::Type::IMU: this->apply_imu(msg->as_imu()); break;
      case SensorMsg::Type::DVL: this->apply_dvl(msg->as_dvl()); break;
      case SensorMsg::Type::DEPTH: this->apply_depth(msg->as_depth()); break;
    }

    Checkpoint c(msg);
    this->ekf.get_state(c.state);
    this->ekf.get_cov(c.cov);
    this->checkpoints.push_back(c);
}

void StateEstimator::apply_imu(const ImuMsg &msg)
{
  double time = msg.stamp.toSec();
  this->ekf.handle_imu_measurement(time, msg.orientation, msg.rate_of_turn, msg.linear_acceleration, this->imu_covariance);
}

void StateEstimator::apply_dvl(const DvlMsg &msg)
{
  double time = msg.stamp.toSec();
  this->ekf.handle_dvl_measurement(time, msg.velocity, msg.avg_beam_std_dev * this->dvl_covariance);
  // this->ekf.handle_depth_measurement(time, -msg.altitute, this->depth_covariance);
}

void StateEstimator::apply_depth(const DepthMsg &msg)
{
  double time = msg.stamp.toSec();
  this->ekf.handle_depth_measurement(time, msg.depth, this->depth_covariance);
}

void StateEstimator::handle_imu(const tauv_msgs::XsensImuData::ConstPtr& msg)
{
  if (!this->is_initialized) {
    this->is_initialized = true;
    this->last_evaluation_time = msg->header.stamp - this->horizon_delay;
  }

  boost::shared_ptr<SensorMsg> sensor_msg = boost::shared_ptr<SensorMsg>(new SensorMsg(msg));

  if (msg->header.stamp < this->last_evaluation_time && this->checkpoints.full()) {
    this->delayed_queue.push(sensor_msg);
  } else {
    this->realtime_queue.push(sensor_msg);
  }
}

void StateEstimator::handle_dvl(const tauv_msgs::TeledyneDvlData::ConstPtr& msg)
{
  if (!this->is_initialized) {
    this->is_initialized = true;
    this->last_evaluation_time = msg->header.stamp - this->horizon_delay;
  }

  // if (!msg->is_hr_velocity_valid) return;
  boost::shared_ptr<SensorMsg> sensor_msg = boost::shared_ptr<SensorMsg>(new SensorMsg(msg));

  if (msg->header.stamp < this->last_evaluation_time && this->checkpoints.full()) {
    this->delayed_queue.push(sensor_msg);
  } else {
    this->realtime_queue.push(sensor_msg);
  }
}

void StateEstimator::handle_depth(const std_msgs::Float32::ConstPtr& msg)
{
  ros::Time time = ros::Time::now();
  if (!this->is_initialized) {
    this->is_initialized = true;
    this->last_evaluation_time = time - this->horizon_delay;
  }

  boost::shared_ptr<SensorMsg> sensor_msg = boost::shared_ptr<SensorMsg>(new SensorMsg(msg));

  if (time < this->last_evaluation_time && this->checkpoints.full()) {
    this->delayed_queue.push(sensor_msg);
  } else {
    this->realtime_queue.push(sensor_msg);
  }
}

bool StateEstimator::handle_set_pose(
  tauv_msgs::SetPose::Request &req,
  tauv_msgs::SetPose::Response &res)
{
    ROS_INFO("handle set pose");

    if (!this->is_initialized) {
        res.success = false;
        return false;
    }

    ros::Time current_time = ros::Time::now() - this->horizon_delay;

    this->checkpoints.clear();
    this->last_evaluation_time = current_time;

    Eigen::Matrix<double, 15, 1> original_state;
    this->ekf.get_state(original_state);
    double original_yaw = original_state(Ekf::StateIndex::YAW);

    Eigen::Matrix<double, 15, 1> state = Eigen::Matrix<double, 15, 1>::Zero();
    state[0] = req.position.x;
    state[1] = req.position.y;
    state[2] = req.position.z;

    Eigen::Matrix<double, 15, 15> cov = 1e-9 * Eigen::Matrix<double, 15, 15>::Identity();

    this->ekf.set_state(state);
    this->ekf.set_cov(cov);
    this->ekf.set_time(current_time.toSec());
    this->ekf.set_reference_yaw(original_yaw);

    this->last_evaluation_time = current_time;

    res.success = true;
    return true;
}

void StateEstimator::publish_navigation_state(
  ros::Time time,
  const Eigen::Vector3d &position,
  const Eigen::Vector3d &velocity,
  const Eigen::Vector3d &acceleration,
  const Eigen::Vector3d &orientation,
  const Eigen::Vector3d &angular_velocity,
  const Eigen::Vector3d &angular_acceleration)
{
  tauv_msgs::NavigationState msg;
  msg.header.stamp = time;
  msg.position.x = position.x();
  msg.position.y = position.y();
  msg.position.z = position.z();
  msg.linear_velocity.x = velocity.x();
  msg.linear_velocity.y = velocity.y();
  msg.linear_velocity.z = velocity.z();
  msg.linear_acceleration.x = acceleration.x();
  msg.linear_acceleration.y = acceleration.y();
  msg.linear_acceleration.z = acceleration.z();
  msg.orientation.x = orientation.x();
  msg.orientation.y = orientation.y();
  msg.orientation.z = orientation.z();
  msg.euler_velocity.x = angular_velocity.x();
  msg.euler_velocity.y = angular_velocity.y();
  msg.euler_velocity.z = angular_velocity.z();
  msg.euler_acceleration.x = angular_acceleration.x();
  msg.euler_acceleration.y = angular_acceleration.y();
  msg.euler_acceleration.z = angular_acceleration.z();
  this->navigation_state_pub.publish(msg);
}

void StateEstimator::publish_odom(
  ros::Time time,
  const Eigen::Vector3d &position,
  const Eigen::Vector3d &velocity,
  const Eigen::Vector3d &orientation,
  const Eigen::Vector3d &angular_velocity)
{
  Eigen::Quaterniond orientation_quat = rpy_to_quat(orientation);
  nav_msgs::Odometry odom_msg;
  odom_msg.header.stamp = time;
  odom_msg.header.frame_id = this->tf_namespace + "/odom";
  odom_msg.child_frame_id = this->tf_namespace + "/vehicle";
  odom_msg.pose.pose.position.x = position.x();
  odom_msg.pose.pose.position.y = position.y();
  odom_msg.pose.pose.position.z = position.z();
  odom_msg.pose.pose.orientation.x = orientation_quat.x();
  odom_msg.pose.pose.orientation.y = orientation_quat.y();
  odom_msg.pose.pose.orientation.z = orientation_quat.z();
  // TODO: CHECK THIS. IT USED TO BE NEGATED
  odom_msg.pose.pose.orientation.w = -orientation_quat.w();
  odom_msg.twist.twist.linear.x = velocity.x();
  odom_msg.twist.twist.linear.y = velocity.y();
  odom_msg.twist.twist.linear.z = velocity.z();
  odom_msg.twist.twist.angular.x = angular_velocity.x();
  odom_msg.twist.twist.angular.y = angular_velocity.y();
  odom_msg.twist.twist.angular.z = angular_velocity.z();
  this->odom_pub.publish(odom_msg);
}

void StateEstimator::publish_tf(
  ros::Time time,
  const Eigen::Vector3d &position,
  const Eigen::Vector3d &orientation)
{
  tf::Transform odom_tf;
  odom_tf.setOrigin(tf::Vector3(position.x(), position.y(), position.z()));
  tf::Quaternion odom_quat;
  odom_quat.setRPY(orientation.x(), orientation.y(), orientation.z());
  odom_tf.setRotation(odom_quat);
  this->odom_tf_broadcaster.sendTransform(tf::StampedTransform(odom_tf, time, this->tf_namespace + "/odom", this->tf_namespace + "/vehicle"));
}

StateEstimator::ImuMsg::ImuMsg(const tauv_msgs::XsensImuData::ConstPtr &msg) {
    this->stamp = msg->header.stamp;
    this->orientation = Eigen::Vector3d { msg->orientation.x, msg->orientation.y, msg->orientation.z };
    this->rate_of_turn = Eigen::Vector3d { msg->rate_of_turn.x, msg->rate_of_turn.y, msg->rate_of_turn.z };
//     Eigen::Vector3d raw_linear_acceleration = Eigen::Vector3d { msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z };
//     Eigen::Vector3d gravity { 0.0, 0.0, -9.806 };
     Eigen::Vector3d free_acceleration = Eigen::Vector3d { msg->free_acceleration.x, msg->free_acceleration.y, msg->free_acceleration.z };
     Eigen::Quaterniond orientation_quat = rpy_to_quat(orientation);
     Eigen::Vector3d body_free_acceleration = orientation_quat.toRotationMatrix() * free_acceleration;
//    Eigen::Vector3d linear_acceleration { 0, 0, 0 };
    this->linear_acceleration = body_free_acceleration;
}

StateEstimator::DvlMsg::DvlMsg(const tauv_msgs::TeledyneDvlData::ConstPtr &msg) {
  this->stamp = msg->header.stamp;
  this->velocity = Eigen::Vector3d { msg->velocity.x, msg->velocity.y, -msg->velocity.z };
  this->avg_beam_std_dev =
    (msg->beam_standard_deviations.elems[0]
     + msg->beam_standard_deviations.elems[1]
     + msg->beam_standard_deviations.elems[2]
     + msg->beam_standard_deviations.elems[3]) / 4.0;
  this->altitute = msg->vertical_range;
}

StateEstimator::DepthMsg::DepthMsg(const std_msgs::Float32::ConstPtr &msg) {
  this->stamp = ros::Time::now();
  this->depth = msg->data;
}
