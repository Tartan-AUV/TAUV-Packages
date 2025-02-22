#include <ros/ros.h>
#include <tf/transform_listener.h>
#include <tf/transform_datatypes.h>
#include <eigen3/Eigen/Dense>
#include "transforms.h"

namespace transforms {
  TransformClient::TransformClient(ros::NodeHandle &n) : tf_listener(n) 
  {
    
  }

  void TransformClient::transform_vector(Eigen::Vector3d &vin, Eigen::Vector3d &vout, Frame from, Frame to)
  {
    tf::Stamped<tf::Vector3> tvin;
    tvin.setValue(vin.x(), vin.y(), vin.z());
    tvin.frame_id_ = this->frame_ids.find(from)->second;
    tvin.stamp_ = ros::Time::now();

    tf::Stamped<tf::Vector3> tvout;
    this->tf_listener.transformVector(this->frame_ids.find(to)->second, ros::Time(0), tvin, this->frame_ids.find(from)->second, tvout);

    vout(0) = tvout.x();
    vout(1) = tvout.y();
    vout(2) = tvout.z();
    return;
  }
}
