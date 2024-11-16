#!/usr/bin/env python3

import rospy
import tf
import tf2_ros
import tf2_geometry_msgs
import math
import geometry_msgs.msg

def test_camera_transform():
    rospy.init_node("test_camera_tf")

    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)

    tf_broadcaster = tf2_ros.TransformBroadcaster()
    
    rate = rospy.Rate(60)

    x, y, theta = 0, 0, 0
    dt = (2 * math.pi / 60) / 2
    radius = 3

    while not rospy.is_shutdown():
        # make one revolution every 2 seconds
        x += radius * math.cos(dt)
        y += radius * math.sin(dt)
        z += math.sin(dt) + 1

        theta += dt

        world_to_vehicle_transform = geometry_msgs.msg.TransformStamped()
        world_to_vehicle_transform.header.frame_id = "world_ned"
        world_to_vehicle_transform.child_frame_id = "vehicle_ned"

        world_to_vehicle_transform.transform.translation.x = x
        world_to_vehicle_transform.transform.translation.y = y
        world_to_vehicle_transform.transform.translation.z = z
        
        quat = tf.transformations.quaternion_from_euler(float(0),float(0),float(theta))

        world_to_vehicle_transform.transform.rotation.x = quat[0]
        world_to_vehicle_transform.transform.rotation.x = quat[1]
        world_to_vehicle_transform.transform.rotation.x = quat[2]
        world_to_vehicle_transform.transform.rotation.x = quat[3]

        tf_broadcaster.sendTransform(world_to_vehicle_transform)

        rate.sleep()


if __name__ == "__main__":
    test_camera_transform()