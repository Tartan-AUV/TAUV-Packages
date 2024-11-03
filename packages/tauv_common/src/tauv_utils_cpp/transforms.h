#include <ros/ros.h>
#include <eigen3/Eigen/Dense>
#include <tf/transform_listener.h>

#pragma once

namespace transforms {
  enum Frame {
    ODOM_NED,
    ODOM_ENU,
    VEHICLE_NED,
    VEHICLE_ENU,
    IMU_NED,
    IMU_ENU,
    DVL_ENU,
    DVL_NED
  };

  class TransformClient {
    public:
      TransformClient(ros::NodeHandle &n);

      void transform_vector(Eigen::Vector3d &vin, Eigen::Vector3d &vout, Frame from, Frame to);

    private:
      tf::TransformListener tf_listener;

      const std::map<Frame, std::string> frame_ids {
        { Frame::ODOM_NED, "odom_ned" },
        { Frame::ODOM_ENU, "odom_enu" },
        { Frame::VEHICLE_NED, "vehicle_ned" },
        { Frame::VEHICLE_ENU, "vehicle_enu" },
        { Frame::IMU_NED, "imu_ned" },
        { Frame::IMU_ENU, "imu_enu" },
        { Frame::DVL_NED, "dvl_ned" },
        { Frame::DVL_ENU, "dvl_enu" },
      };
  };
}
