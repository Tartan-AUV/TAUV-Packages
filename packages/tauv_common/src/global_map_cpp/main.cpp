#include <iostream>
#include <ros/ros.h>
#include "./global_map.h"

int main(int argc, char *argv[])
{
  cout<<"HERE!!!\n";
  ros::init(argc, argv, "global_map");
  ros::NodeHandle n;

  ROS_INFO("Launched!");

  GlobalMap s(n);
  ROS_INFO("Launched!");

  while (ros::ok())
  {
    ros::spinOnce();
  }

  return 0;
}