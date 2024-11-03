#include <mutex>
#include <set>
#include <vector>
#include <ros/ros.h>
#include <tauv_msgs/AlarmReport.h>
#include "alarms.h"

#pragma once

namespace tauv_alarms {
  class AlarmClient {
      public:
          AlarmClient(ros::NodeHandle& n);
          void set(tauv_alarms::AlarmType type, const std::string &msg, bool value = true);
          void clear(tauv_alarms::AlarmType type, const std::string &msg);
          bool check(tauv_alarms::AlarmType type);
      private:
          ros::NodeHandle &n;

          std::set<tauv_alarms::AlarmType> active_alarms;

          ros::Time last_update_time;
          ros::Duration timeout;

          std::mutex lock;

          ros::Subscriber report_sub;
          ros::ServiceClient sync_srv;

          void handle_report(const tauv_msgs::AlarmReport::ConstPtr &msg);

          void set_active_alarms(std::vector<int> alarms);
  };
}
