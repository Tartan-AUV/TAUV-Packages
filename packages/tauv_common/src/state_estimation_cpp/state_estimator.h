#include <array>
#include <boost/circular_buffer.hpp>
#include <boost/heap/priority_queue.hpp>
#include <boost/variant.hpp>
#include <eigen3/Eigen/Dense>
#include <mutex>
#include <nav_msgs/Odometry.h>
#include <numeric>
#include <ros/ros.h>
#include <tauv_alarms/alarms.h>
#include <tauv_alarms/alarm_client.h>
#include <tauv_msgs/NavigationState.h>
#include <std_msgs/Float32.h>
#include <tauv_msgs/TeledyneDvlData.h>
#include <tauv_msgs/XsensImuData.h>
#include <tf/transform_broadcaster.h>
#include <tauv_msgs/SetPose.h>
#include "ekf.h"

#pragma once

class StateEstimator {
  public:
    StateEstimator(ros::NodeHandle& n, ros::NodeHandle &pn);

    void update(const ros::TimerEvent& e);

    void handle_imu(const tauv_msgs::XsensImuData::ConstPtr& msg);
    void handle_dvl(const tauv_msgs::TeledyneDvlData::ConstPtr& msg);
    void handle_depth(const std_msgs::Float32::ConstPtr& msg);

    class ImuMsg;
    class DvlMsg;
    class DepthMsg;
    class SensorMsg;
    class Checkpoint;

  private:
    ros::NodeHandle& n;
    ros::NodeHandle& pn;
    ros::Timer timer;

    std::string tf_namespace;

    tauv_alarms::AlarmClient alarm_client;

    ros::Subscriber imu_sub;
    ros::Subscriber dvl_sub;
    ros::Subscriber depth_sub;

    ros::Publisher navigation_state_pub;
    ros::Publisher odom_pub;

    ros::ServiceServer set_pose_srv;

    tf::TransformBroadcaster odom_tf_broadcaster;

    Ekf ekf;

    bool is_initialized;

    ros::Time last_evaluation_time;
    Eigen::Vector3d last_angular_velocity;

    boost::circular_buffer<Checkpoint> checkpoints;

    boost::heap::priority_queue<boost::shared_ptr<SensorMsg>, boost::heap::compare<std::greater_equal<boost::shared_ptr<SensorMsg>>>> realtime_queue;
    boost::heap::priority_queue<boost::shared_ptr<SensorMsg>, boost::heap::compare<std::greater_equal<boost::shared_ptr<SensorMsg>>>> delayed_queue;

    ros::Duration dt;
    int max_delayed_queue_size;
    ros::Duration horizon_delay;
    Eigen::Vector3d dvl_offset;
    Eigen::Matrix<double, 15, 1> process_covariance;
    Eigen::Matrix<double, 9, 1> imu_covariance;
    Eigen::Vector3d dvl_covariance;
    double depth_covariance;
    double euler_acceleration_filter_constant;

    Eigen::Vector3d previous_angular_acceleration;

    void load_config();

    void apply_msg(boost::shared_ptr<SensorMsg> msg);
    void apply_imu(const ImuMsg &msg);
    void apply_dvl(const DvlMsg &msg);
    void apply_depth(const DepthMsg &msg);

    bool handle_set_pose(tauv_msgs::SetPose::Request &req, tauv_msgs::SetPose::Response &res);

    void publish_navigation_state(
        ros::Time time,
        const Eigen::Vector3d &position,
        const Eigen::Vector3d &velocity,
        const Eigen::Vector3d &acceleration,
        const Eigen::Vector3d &orientation,
        const Eigen::Vector3d &angular_velocity,
        const Eigen::Vector3d &angular_acceleration);
    void publish_odom(
        ros::Time time,
        const Eigen::Vector3d &position,
        const Eigen::Vector3d &velocity,
        const Eigen::Vector3d &orientation,
        const Eigen::Vector3d &angular_velocity);
    void publish_tf(
        ros::Time time,
        const Eigen::Vector3d &position,
        const Eigen::Vector3d &orientation);
};

class StateEstimator::ImuMsg {
public:
  ros::Time stamp; 
  Eigen::Vector3d orientation;
  Eigen::Vector3d rate_of_turn;
  Eigen::Vector3d linear_acceleration;

  ImuMsg(const tauv_msgs::XsensImuData::ConstPtr &msg);
};

class StateEstimator::DvlMsg {
public:
  ros::Time stamp;
  Eigen::Vector3d velocity;
  double avg_beam_std_dev;
  double altitute;

  DvlMsg(const tauv_msgs::TeledyneDvlData::ConstPtr &msg);
};

class StateEstimator::DepthMsg {
public:
  ros::Time stamp;
  double depth;

  DepthMsg(const std_msgs::Float32::ConstPtr &msg);
};

class StateEstimator::SensorMsg {
public:
  enum Type { IMU, DVL, DEPTH };

  Type type;
  ros::Time stamp;
  boost::variant<ImuMsg, DvlMsg, DepthMsg> msg;

  SensorMsg(const tauv_msgs::XsensImuData::ConstPtr &msg) : msg(msg) {
    this->type = Type::IMU;
    this->stamp = msg->header.stamp;
  }

  SensorMsg(const tauv_msgs::TeledyneDvlData::ConstPtr &msg) : msg(msg) {
    this->type = Type::DVL;
    this->stamp = msg->header.stamp;
  }

  SensorMsg(const std_msgs::Float32::ConstPtr &msg) : msg(msg) {
    this->type = Type::DEPTH;
    this->stamp = ros::Time::now();
//    this->stamp = msg->header.stamp;
  }

  const ImuMsg &as_imu() {
    return boost::get<ImuMsg>(this->msg);
  };

  const DvlMsg &as_dvl() {
    return boost::get<DvlMsg>(this->msg);
  };

  const DepthMsg &as_depth() {
    return boost::get<DepthMsg>(this->msg);
  };
};

bool operator < (const StateEstimator::SensorMsg &lhs, const StateEstimator::SensorMsg &rhs) 
{
  return lhs.stamp < rhs.stamp;
}

bool operator >= (const StateEstimator::SensorMsg &lhs, const StateEstimator::SensorMsg &rhs) 
{
  return lhs.stamp >= rhs.stamp;
}

bool operator < (const boost::shared_ptr<StateEstimator::SensorMsg> &lhs, const boost::shared_ptr<StateEstimator::SensorMsg> &rhs) 
{
  return lhs->stamp < rhs->stamp;
}

bool operator >= (const boost::shared_ptr<StateEstimator::SensorMsg> &lhs, const boost::shared_ptr<StateEstimator::SensorMsg> &rhs) 
{
  return lhs->stamp >= rhs->stamp;
}

class StateEstimator::Checkpoint {
  public:
    boost::shared_ptr<StateEstimator::SensorMsg> msg;
    Eigen::Matrix<double, 15, 1> state;
    Eigen::Matrix<double, 15, 15> cov;

    Checkpoint(boost::shared_ptr<StateEstimator::SensorMsg> msg) : msg(msg) {};
};
