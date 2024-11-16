#include <iostream>
#include <ros/ros.h>
#include "state_estimator.h"

int main(int argc, char *argv[])
{
  ros::init(argc, argv, "state_estimation");
  ros::NodeHandle n;
  ros::NodeHandle pn("~");

  ROS_INFO("Launched!");

  StateEstimator s(n, pn);

  while (ros::ok())
  {
    ros::spinOnce();
  }

  return 0;
}
