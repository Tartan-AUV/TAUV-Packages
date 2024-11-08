
# TAUV_Vehicle
This package is home to vehicle-specific launchfiles and hardware drivers

# LaunchFiles:

## albatross_vehicle
This launchfile launches all necessary vehicle abstraction nodes for Albatross
Namespace: albatross
Nodes: imu_manager, depth_sensor, default_joint_pub, static_tfs, actuators.

# Nodes:

## imu_manager
Nodelet which includes `ImuFilterNodelet` and `PhidgetsImuNodelet`
This nodelet manages the Phidgets IMU and runs the data through a madgwick imu filter
The IMU manager publishes a tf transform between `/odom` and `base_link`, although this should be disabled and replaced with a proper state estimation node that publishes that transform after filtering.

**Requirements:**
The phidget should be plugged into the vehicle, or this node will fail to function.

**Launchfile Parameters:**
- `PhidgetsImuNodelet/frame_id` the tf frame to publish IMU data in. Currently set to `albatross/imu_link`.
- `PhidgetsImuNodelet/period` The publishing rate. Currently 4ms (250Hz)
- `ImuFilterNodelet/use_mag` boolean on whether to use the magnetometer or not. True for now.
- `ImuFilterNodelet/fixed_frame` defaults to `odom`

**Publishes:**
- `imu/data_raw` raw IMU data from the phidget
- `imu/mag` raw magnetometer data from the phidget
- `imu/data` pose data from the madgwick filter, in the frame specified.
- `/diagnostics` diagnostic info from the phidget
(Also a bunch of other topics that I won't document here)

## depth_sensor
This node wraps the Blue Robotics depth sensor driver.

Requirements:
python-smbus, libi2c-dev
Depth sensor must be plugged into i2c bus 0 on the robot.

**External Parameters:** 
- `/vehicle_params/has_depth_sensor` must be set to true
- `/vehicle_params/depth_sensor` should be either 'bar30' or 'bar02'

**Publishes:**
- `depth` the depth of the robot in meters. TODO: make this timestamped
- `temperature` the water temperature

## actuators
The actuators node interfaces with the Pololu Maestro pwm module to control servos and ESCs. Currently, it only supports setups with 8 thrusters and no servos. Servo and variable thruster support is TODO.

**External Parameters**
- `/vehicle_params/maestro_tty` Path to the tty bus for the maestro. by-id paths are preferred for consistency.
- `/vehicle_params/maestro_thruster` Bool indicating thruster support
- `/vehicle_params/meastro_servos` Bool indicating servo support
- `/vehicle_params/maestro_thruster_channels` The thruster channels to use, in thruster order (not channel order)
- `/vehicle_params/meastro_servo_channels` The servo channels to use
- `/vehicle_params/thruster_timeout_s` Allowable timeout between thruster commands before disarming
- `/vehicle_params/maestro_inverted_thrusters` Array of 0s/1s to indicate which thrusters are inverted in hw
- `/vehicle_params/esc_pwm_reverse` value of full reverse pwm (in uS)
- `/vehicle_params/esc_pwm_forwards` value of full forwards pwm (in uS)

**Subscribes**

 - `thrusters/{n}/input` The normalized input (-1 to 1) for thruster n

## joint_state_pub
Very simple module that publishes *default* joint states in order to allow robot_state_publisher to publish tf frames from the URDF. This node can be configured to listen to actual joint states and republish them with the joint state array. The default joints in the URDF are all unused but also unfixed, so this node addresses that. It will be useful if we mean to attach arms to the robot.

See: http://wiki.ros.org/joint_state_publisher

**external parameters**
 - `robot_description` URDF string with the robot description

**published topics**
- `joint_states` The joint state list used by robot_state_publisher

## static_tfs
The launch file static_tfs.launch is able to launch a large list of static transforms. Currently, no static transforms are necessary but there is a list of examples that republish certain frames with new names.

