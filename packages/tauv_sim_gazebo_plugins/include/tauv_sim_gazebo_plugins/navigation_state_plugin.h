#pragma once

#include <ros/ros.h>
#include <tf2_ros/transform_broadcaster.h>
#include <boost/scoped_ptr.hpp>

#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/common.hh>
#include <ignition/math/Vector3.hh>
#include <ignition/math/Pose3.hh>

#include <tauv_msgs/NavigationState.h>

namespace gazebo {
class NavigationStatePlugin : public ModelPlugin
{
  public:
  void Load(physics::ModelPtr model, sdf::ElementPtr sdf);
  void OnUpdate(const common::UpdateInfo& info);

  private:
  event::ConnectionPtr updateConnection;

  boost::scoped_ptr<ros::NodeHandle> rosNode;

  physics::ModelPtr model;
  physics::LinkPtr link;

  ros::Publisher pubNavState;
  ros::Publisher pubOdom;

  tf2_ros::TransformBroadcaster odomVehicleTfBroadcaster;

  common::Time publishPeriod;
  common::Time lastPublishTime;
  ignition::math::Vector3d lastEulerVel;

  std::string tfNamespace;
};
}
