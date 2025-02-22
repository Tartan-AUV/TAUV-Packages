#include <ros/ros.h>
#include <tf2_ros/transform_broadcaster.h>
#include <boost/scoped_ptr.hpp>

#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/common.hh>
#include <ignition/math/Vector3.hh>
#include <ignition/math/Pose3.hh>
#include <ignition/math/Quaternion.hh>
#include <ignition/math/Matrix4.hh>

#include <tauv_msgs/NavigationState.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/TransformStamped.h>

#include <tauv_sim_gazebo_plugins/navigation_state_plugin.h>

GZ_REGISTER_MODEL_PLUGIN(gazebo::NavigationStatePlugin)

namespace gazebo {
void NavigationStatePlugin::Load(physics::ModelPtr model, sdf::ElementPtr sdf)
{
  if (!ros::isInitialized())
  {
    gzerr << "ROS not initialized." << std::endl;
    return;
  }

  this->model = model;

  GZ_ASSERT(sdf->HasElement("linkName"), "SDF missing linkName.");
  std::string linkName = sdf->Get<std::string>("linkName");
  this->link = this->model->GetLink(linkName);

  GZ_ASSERT(sdf->HasElement("publishRate"), "SDF missing publishRate.");
  this->publishPeriod = 1.0 / (double)sdf->Get<int>("publishRate");

  GZ_ASSERT(sdf->HasElement("nodeName"), "SDF missing nodeName.");
  std::string nodeName = sdf->Get<std::string>("nodeName");

  GZ_ASSERT(sdf->HasElement("navigationStateTopic"), "SDF missing navigationStateTopic.");
  std::string navigationStateTopic = sdf->Get<std::string>("navigationStateTopic");

  GZ_ASSERT(sdf->HasElement("odomTopic"), "SDF missing odomTopic.");
  std::string odomTopic = sdf->Get<std::string>("odomTopic");

  GZ_ASSERT(sdf->HasElement("tfNamespace"), "SDF missing tfNamespace.");
  this->tfNamespace = sdf->Get<std::string>("tfNamespace");

  this->rosNode.reset(new ros::NodeHandle(nodeName));

  this->pubNavState = this->rosNode->advertise<tauv_msgs::NavigationState>(
    navigationStateTopic, 10
  );

  this->pubOdom = this->rosNode->advertise<nav_msgs::Odometry>(
    odomTopic, 10
  );

  this->updateConnection = event::Events::ConnectWorldUpdateBegin(
    boost::bind(&NavigationStatePlugin::OnUpdate, this, _1)
  );
}

void NavigationStatePlugin::OnUpdate(const common::UpdateInfo& info)
{
  if (info.simTime - this->lastPublishTime < this->publishPeriod) {
    return;
  }

  ignition::math::Pose3d linkPose = this->link->WorldPose();
  ignition::math::Vector3d linkLinVelENU = this->link->WorldLinearVel();
  ignition::math::Vector3d linkLinVel(
    linkLinVelENU.X(),
    -linkLinVelENU.Y(),
    -linkLinVelENU.Z()
  );
  ignition::math::Vector3d linkAngVelENU = this->link->WorldAngularVel();
  ignition::math::Vector3d linkAngVel(
    linkAngVelENU.X(),
    -linkAngVelENU.Y(),
    -linkAngVelENU.Z()
  );

  ignition::math::Quaterniond rot(
    linkPose.Rot().W(),
    linkPose.Rot().X(),
    -linkPose.Rot().Y(),
    -linkPose.Rot().Z()
  );

  ignition::math::Vector3d linkLinAccelENU = this->link->WorldLinearAccel();
  ignition::math::Vector3d linkLinAccel(
    linkLinAccelENU.X(),
    -linkLinAccelENU.Y(),
    -linkLinAccelENU.Z()
  );

  ignition::math::Vector3d linVel = rot.Inverse() * linkLinVel;
  ignition::math::Vector3d linAccel = rot.Inverse() * linkLinAccel;

  ignition::math::Matrix3d angVelToEulerVel(
    1, sin(rot.Roll()) * tan(rot.Pitch()), cos(rot.Roll()) * tan(rot.Pitch()),
    0, cos(rot.Roll()), -sin(rot.Roll()),
    0, sin(rot.Roll()) / cos(rot.Pitch()), cos(rot.Roll()) / cos(rot.Pitch())
  );

  ignition::math::Vector3d angVel = rot.Inverse() * linkAngVel;
  ignition::math::Vector3d eulerVel = angVelToEulerVel * (rot.Inverse() * linkAngVel);

  ignition::math::Vector3d eulerAccel = (eulerVel - this->lastEulerVel) / (info.simTime - this->lastPublishTime).Double();

  tauv_msgs::NavigationState navStateMsg;
  navStateMsg.position.x = linkPose.Pos().X();
  navStateMsg.position.y = -linkPose.Pos().Y();
  navStateMsg.position.z = -linkPose.Pos().Z();

  navStateMsg.orientation.x = rot.Roll();
  navStateMsg.orientation.y = rot.Pitch();
  navStateMsg.orientation.z = rot.Yaw();

  navStateMsg.linear_velocity.x = linVel.X();
  navStateMsg.linear_velocity.y = linVel.Y();
  navStateMsg.linear_velocity.z = linVel.Z();

  navStateMsg.linear_acceleration.x = linAccel.X();
  navStateMsg.linear_acceleration.y = linAccel.Y();
  navStateMsg.linear_acceleration.z = linAccel.Z();

  navStateMsg.euler_velocity.x = eulerVel.X();
  navStateMsg.euler_velocity.y = eulerVel.Y();
  navStateMsg.euler_velocity.z = eulerVel.Z();

  navStateMsg.euler_acceleration.x = eulerAccel.X();
  navStateMsg.euler_acceleration.y = eulerAccel.Y();
  navStateMsg.euler_acceleration.z = eulerAccel.Z();

  this->pubNavState.publish(navStateMsg);

  ros::Time current_time = ros::Time::now();

  nav_msgs::Odometry odomMsg;
  odomMsg.header.stamp = current_time;
  odomMsg.header.frame_id = this->tfNamespace + "/odom";
  odomMsg.child_frame_id = this->tfNamespace + "/vehicle";
  odomMsg.pose.pose.position.x = navStateMsg.position.x;
  odomMsg.pose.pose.position.y = navStateMsg.position.y;
  odomMsg.pose.pose.position.z = navStateMsg.position.z;
  odomMsg.pose.pose.orientation.x = rot.X();
  odomMsg.pose.pose.orientation.y = rot.Y();
  odomMsg.pose.pose.orientation.z = rot.Z();
  odomMsg.pose.pose.orientation.w = rot.W();
  odomMsg.twist.twist.linear = navStateMsg.linear_velocity;
  odomMsg.twist.twist.angular.x = angVel.X();
  odomMsg.twist.twist.angular.x = angVel.Y();
  odomMsg.twist.twist.angular.x = angVel.Z();
  this->pubOdom.publish(odomMsg);

  geometry_msgs::TransformStamped odomVehicleTf;
  odomVehicleTf.header.stamp = current_time;
  odomVehicleTf.header.frame_id = this->tfNamespace + "/odom";
  odomVehicleTf.child_frame_id = this->tfNamespace + "/vehicle";
  odomVehicleTf.transform.translation.x = navStateMsg.position.x;
  odomVehicleTf.transform.translation.y = navStateMsg.position.y;
  odomVehicleTf.transform.translation.z = navStateMsg.position.z;
  odomVehicleTf.transform.rotation.x = rot.X();
  odomVehicleTf.transform.rotation.y = rot.Y();
  odomVehicleTf.transform.rotation.z = rot.Z();
  odomVehicleTf.transform.rotation.w = rot.W();
  this->odomVehicleTfBroadcaster.sendTransform(odomVehicleTf);

  this->lastPublishTime = info.simTime;
  this->lastEulerVel = eulerVel;
}
}
