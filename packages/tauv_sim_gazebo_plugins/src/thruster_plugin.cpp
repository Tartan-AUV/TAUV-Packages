#include <boost/bind.hpp>
#include <ros/ros.h>
#include <boost/scoped_ptr.hpp>
#include <std_msgs/Float64.h>

#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/common.hh>
#include <ignition/math/Vector3.hh>

#include <tauv_sim_gazebo_plugins/thruster_plugin.h>

GZ_REGISTER_MODEL_PLUGIN(gazebo::ThrusterPlugin)

namespace gazebo {
void ThrusterPlugin::Load(physics::ModelPtr model, sdf::ElementPtr sdf)
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

  GZ_ASSERT(sdf->HasElement("timeConstant"), "SDF missing timeConstant.");
  this->tau = sdf->Get<double>("timeConstant");

  GZ_ASSERT(sdf->HasElement("maxThrust"), "SDF missing maxThrust.");
  this->maxThrust = sdf->Get<double>("maxThrust");

  GZ_ASSERT(sdf->HasElement("minThrust"), "SDF missing minThrust.");
  this->minThrust = sdf->Get<double>("minThrust");

  GZ_ASSERT(sdf->HasElement("publishRate"), "SDF missing publishRate.");
  this->publishPeriod = 1.0 / (double)sdf->Get<int>("publishRate");

  GZ_ASSERT(sdf->HasElement("nodeName"), "SDF missing nodeName.");
  std::string nodeName = sdf->Get<std::string>("nodeName");

  GZ_ASSERT(sdf->HasElement("targetThrustTopic"), "SDF missing targetThrustTopic.");
  std::string targetThrustTopic = sdf->Get<std::string>("targetThrustTopic");

  GZ_ASSERT(sdf->HasElement("thrustTopic"), "SDF missing thrustTopic.");
  std::string thrustTopic = sdf->Get<std::string>("thrustTopic");

  this->rosNode.reset(new ros::NodeHandle(nodeName));

  this->subTargetThrust = this->rosNode->subscribe<std_msgs::Float64>(
    targetThrustTopic,
    10,
    boost::bind(&ThrusterPlugin::HandleTargetThrust, this, _1)
  );

  this->pubThrust = this->rosNode->advertise<std_msgs::Float64>(
    thrustTopic, 10
  );

  this->updateConnection = event::Events::ConnectWorldUpdateBegin(
    boost::bind(&ThrusterPlugin::OnUpdate, this, _1)
  );
}

void ThrusterPlugin::OnUpdate(const common::UpdateInfo& info)
{
  double dt = info.simTime.Double() - this->lastUpdateTime;

  double alpha = std::exp(-dt / this->tau);

  this->thrust = alpha * this->thrust + (1.0 - alpha) * this->targetThrust;
  this->thrust = std::min(this->thrust, this->maxThrust);
  this->thrust = std::max(this->thrust, this->minThrust);

  this->lastUpdateTime = info.simTime.Double();

  ignition::math::Vector3d force(this->thrust, 0, 0);
  this->link->AddRelativeForce(force);

  if (info.simTime - this->lastPublishTime >= this->publishPeriod) {
    std_msgs::Float64 thrustMsg;
    thrustMsg.data = this->thrust;
    this->pubThrust.publish(thrustMsg);

    this->lastPublishTime = info.simTime;
  }
}

void ThrusterPlugin::HandleTargetThrust(const std_msgs::Float64::ConstPtr& msg)
{
  this->targetThrust = msg->data;
}
}
