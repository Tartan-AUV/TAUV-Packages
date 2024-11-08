#include <mutex>
#include <set>
#include <vector>
#include <ros/ros.h>
#include <tauv_msgs/AlarmReport.h>
#include <tauv_msgs/AlarmWithMessage.h>
#include <tauv_msgs/SyncAlarms.h>
#include "alarm_client.h"
#include "alarms.h"

namespace tauv_alarms {
  AlarmClient::AlarmClient(ros::NodeHandle &n) : n(n)
  {
    this->timeout = ros::Duration(1.0);

    this->active_alarms = std::set<tauv_alarms::AlarmType> { tauv_alarms::AlarmType::UNKNOWN_ALARMS };

    this->report_sub = n.subscribe("alarms/report", 100, &AlarmClient::handle_report, this);
    this->sync_srv = n.serviceClient<tauv_msgs::SyncAlarms>("alarms/sync");
  }

  void AlarmClient::set(tauv_alarms::AlarmType type, const std::string &msg, bool value)
  {
    this->lock.lock();

    bool update = false;
    if (value) {
      this->active_alarms.insert(type);
      this->last_update_time = ros::Time::now();
      update = true;
    } else {
      this->active_alarms.erase(type);
      this->last_update_time = ros::Time::now();
      update = true;
    }
    this->lock.unlock();

    if (!update) return;

    tauv_msgs::AlarmWithMessage alarm;
    alarm.id = type;
    alarm.message = msg;
    alarm.set = value;

    tauv_msgs::SyncAlarms srv;
    srv.request.diff = std::vector<tauv_msgs::AlarmWithMessage> { alarm };

    bool success = this->sync_srv.call(srv);
    if (!success) return;

    this->lock.lock();
    this->last_update_time = srv.response.stamp;
    this->set_active_alarms(srv.response.active_alarms);
    this->lock.unlock();
  }

  void AlarmClient::clear(tauv_alarms::AlarmType type, const std::string &msg)
  {
    this->set(type, msg, false);
  }

  bool AlarmClient::check(tauv_alarms::AlarmType type)
  {
    if (ros::Time::now() - this->last_update_time > this->timeout) {
      return type == tauv_alarms::AlarmType::UNKNOWN_ALARMS;
    }

    return this->active_alarms.find(type) != this->active_alarms.end();
  }

  void AlarmClient::handle_report(const tauv_msgs::AlarmReport::ConstPtr &msg)
  {
    this->lock.lock();
    this->last_update_time = ros::Time::now();
    this->set_active_alarms(msg->active_alarms);
    this->lock.unlock();
  }

  void AlarmClient::set_active_alarms(std::vector<int> alarms) 
  {
    this->active_alarms.clear();
    std::transform(alarms.begin(), alarms.end(), std::inserter(this->active_alarms, this->active_alarms.end()), [](int alarm) -> tauv_alarms::AlarmType { return static_cast<tauv_alarms::AlarmType>(alarm); });
  }
}

