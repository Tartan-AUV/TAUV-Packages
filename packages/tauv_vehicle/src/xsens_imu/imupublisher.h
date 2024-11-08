
//  Copyright (c) 2003-2020 Xsens Technologies B.V. or subsidiaries worldwide.
//  All rights reserved.
//
//  Redistribution and use in source and binary forms, with or without modification,
//  are permitted provided that the following conditions are met:
//
//  1.	Redistributions of source code must retain the above copyright notice,
//  	this list of conditions, and the following disclaimer.
//
//  2.	Redistributions in binary form must reproduce the above copyright notice,
//  	this list of conditions, and the following disclaimer in the documentation
//  	and/or other materials provided with the distribution.
//
//  3.	Neither the names of the copyright holders nor the names of their contributors
//  	may be used to endorse or promote products derived from this software without
//  	specific prior written permission.
//
//  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
//  EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
//  MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL
//  THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
//  SPECIAL, EXEMPLARY OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
//  OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
//  HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY OR
//  TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
//  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.THE LAWS OF THE NETHERLANDS
//  SHALL BE EXCLUSIVELY APPLICABLE AND ANY DISPUTES SHALL BE FINALLY SETTLED UNDER THE RULES
//  OF ARBITRATION OF THE INTERNATIONAL CHAMBER OF COMMERCE IN THE HAGUE BY ONE OR MORE
//  ARBITRATORS APPOINTED IN ACCORDANCE WITH SAID RULES.
//

#ifndef IMUPUBLISHER_H
#define IMUPUBLISHER_H

#include "packetcallback.h"
#include <math.h>
#include <tauv_msgs/XsensImuData.h>

struct ImuPublisher : public PacketCallback
{
    ros::Publisher data_pub;

    ImuPublisher(ros::NodeHandle &node)
    {
        int pub_queue_size = 100;
        data_pub = node.advertise<tauv_msgs::XsensImuData>("vehicle/xsens_imu/raw_data", pub_queue_size);
    }

    void operator()(const XsDataPacket &packet, ros::Time timestamp)
    {
        if (!packet.containsSampleTimeFine())
          return;

        const uint32_t SAMPLE_TIME_FINE_HZ = 10000UL;
        const uint32_t ONE_GHZ = 1000000000UL;
        uint32_t sec, nsec, t_fine;

        t_fine = packet.sampleTimeFine();
        sec = t_fine / SAMPLE_TIME_FINE_HZ;
        nsec = (t_fine % SAMPLE_TIME_FINE_HZ) * (ONE_GHZ / SAMPLE_TIME_FINE_HZ);

        if (packet.containsSampleTimeCoarse())
        {
            sec = packet.sampleTimeCoarse();
        }

        ros::Time sample_time(sec, nsec);
        
        bool orientation_available = packet.containsOrientation();
        bool rate_of_turn_available = packet.containsCalibratedGyroscopeData();
        bool linear_acceleration_available = packet.containsCalibratedAcceleration();
        bool free_acceleration_available = packet.containsFreeAcceleration();

        geometry_msgs::Vector3 orientation;
        if (orientation_available)
        {
            // Output in degrees (line 124 `xsdataidentifier.h`)
            XsEuler a = packet.orientationEuler();

            orientation.x = a.roll() * (M_PI / 180.0);
            orientation.y = a.pitch() * (M_PI / 180.0);
            orientation.z = a.yaw() * (M_PI / 180.0);
        }

        geometry_msgs::Vector3 rate_of_turn;
        if (rate_of_turn_available)
        {
            // Output in rad/s (line 152 `xsdataidentifier.h`)
            XsVector a = packet.calibratedGyroscopeData();

            rate_of_turn.x = a[0];
            rate_of_turn.y = a[1];
            rate_of_turn.z = a[2];
        }

        geometry_msgs::Vector3 linear_acceleration;
        if (linear_acceleration_available)
        {
            // Output in m/s^2 (line 131 `xsdataidentifier.h`)
            XsVector a = packet.calibratedAcceleration();

            linear_acceleration.x = a[0];
            linear_acceleration.y = a[1];
            linear_acceleration.z = a[2];
        }

        geometry_msgs::Vector3 free_acceleration;
        if (free_acceleration_available)
        {
            // Output in m/s^2 (line 132 `xsdataidentifier.h`)
            XsVector a = packet.freeAcceleration();

            free_acceleration.x = a[0];
            free_acceleration.y = a[1];
            free_acceleration.z = a[2];
        }

        uint32_t status = packet.status();
        bool triggered_dvl = (status >> 22) & 1;

        if (orientation_available && rate_of_turn_available && free_acceleration_available)
        {
            tauv_msgs::XsensImuData data_msg;

            data_msg.header.stamp = timestamp;

            data_msg.ros_time = timestamp;
            data_msg.imu_time = sample_time;

            data_msg.triggered_dvl = triggered_dvl;

            data_msg.orientation = orientation;
            data_msg.rate_of_turn = rate_of_turn;
            data_msg.linear_acceleration = linear_acceleration;
            data_msg.free_acceleration = free_acceleration;

            data_pub.publish(data_msg);
        }
    }
};

#endif
