#pragma once

#include <ros/ros.h>
#include <boost/scoped_ptr.hpp>
#include <std_msgs/Float64.h>

#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/common.hh>

namespace gazebo
{
class ThrusterPlugin : public ModelPlugin
{
  public:
  void Load(physics::ModelPtr model, sdf::ElementPtr sdf);
  void OnUpdate(const common::UpdateInfo &info);

  private:
  void HandleTargetThrust(const std_msgs::Float64::ConstPtr& target_thrust);

  physics::ModelPtr model;
  physics::LinkPtr link;
  int thrusterID;

  event::ConnectionPtr updateConnection;

  boost::scoped_ptr<ros::NodeHandle> rosNode;

  ros::Subscriber subTargetThrust;
  ros::Publisher pubThrust;

  double tau;
  double maxThrust;
  double minThrust;
  double targetThrust;
  double thrust;
  double lastUpdateTime;

  common::Time publishPeriod;
  common::Time lastPublishTime;
};
}