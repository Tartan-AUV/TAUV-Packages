#!/usr/bin/env python

# This class listens on each /{model_name}/thruster{n}/input topic and sends the
# commanded thrust to the thrusters. Thruster n is mapped to the ESC channel
# according to the thrusters array in the vehicle_params yaml. Ie, input from
# topic n is sent to the channel thrusters[n].
#
# Author: Tom Scherlis 2019
#
# TODO: add servo support

from maestro import Maestro
import rospy
import message_filters
from uuv_gazebo_ros_plugins_msgs.msg import FloatStamped
from std_srvs.srv import SetBool, SetBoolResponse


class ActuatorController:
    def __init__(self):
        print("starting ActuatorController")
        maestro_tty = rospy.get_param('/vehicle_params/maestro_tty')

        self.has_thrusters = rospy.get_param('/vehicle_params/maestro_thrusters')
        self.has_servos = rospy.get_param('/vehicle_params/maestro_servos')

        self.arm_service = rospy.Service("/arm", SetBool, self.srv_arm)

        if not self.has_thrusters and not self.has_servos:
            raise ValueError('''Error: maestro node is running, but vehicle is not
            configured to use it for servos or thrusters.''')

        while not rospy.is_shutdown():
            try:
                self.maestro = Maestro(ttyStr=maestro_tty)
            except:
                print("Could not configure maestro! Trying again in 3 seconds")
                rospy.sleep(3)
                continue
            break
        if rospy.is_shutdown():
            return

        self.thrusters = rospy.get_param('/vehicle_params/maestro_thruster_channels')
        self.servos = rospy.get_param('/vehicle_params/maestro_servo_channels')
        self.timeout = rospy.Duration.from_sec(rospy.get_param('/vehicle_params/thruster_timeout_s'))

        if self.has_servos:
            # TODO: figure out a topic for servo messages and add support
            raise ValueError('Error: Servo support is TODO')

        if self.has_thrusters:
            if len(self.thrusters) != 8:
                raise ValueError('Error: Thruster driver only supports configurations with 8 thrusters')

            self.armed = False

            # thruster command is a dict mapping from  channel to value
            self.thruster_command = {}
            for channel in self.thrusters:
                self.thruster_command[channel] = 0

            # thruster_inversions is a dict mapping from channel to -1 or 1
            inversions = rospy.get_param('/vehicle_params/maestro_inverted_thrusters')
            self.thruster_inversions = {}
            for i, val in enumerate(inversions):
                self.thruster_inversions[self.thrusters[i]] = 1 if val == 0 else -1

            # last thruster message keeps track of the last command, to enforce
            # timeouts. Timing out will cause the driver to send zeros.
            self.last_thruster_msg = None

            # PWM range can be configured
            self.pwm_reverse = rospy.get_param('/vehicle_params/esc_pwm_reverse')
            self.pwm_forward = rospy.get_param('/vehicle_params/esc_pwm_forwards')
            if self.pwm_reverse > self.pwm_forward:
                raise ValueError('Reverse PWM must be less than forward PWM')

            # subscribe to each thruster input topic, and combine them in an
            # approximate time synchronizer.
            topics = []
            subscribers = []
            for i in range(8):
                topicname = 'thrusters/' + str(i) + '/input'
                topics.append(topicname)
                subscribers.append(message_filters.Subscriber(topicname, FloatStamped))
            ts = message_filters.TimeSynchronizer(subscribers, 100)
            ts.registerCallback(self.thruster_callback)

    def start(self):
        r = rospy.Rate(50)  # 50 Hz
        while not rospy.is_shutdown():
            if self.has_thrusters:
                if self.last_thruster_msg is None \
                        or rospy.get_rostime() - self.last_thruster_msg > self.timeout \
                        or not self.armed:
                    # timed out, reset thrusters
                    self.thruster_command = [0] * len(self.thrusters)

                for channel in self.thrusters:
                    cmd = self.speed_to_pwm(self.thruster_command[channel], channel)
                    # command should be in quarter microseconds
                    # send command to maestro ESC board
                    self.maestro.setTarget(cmd*4, channel)
            r.sleep()

    # this callback gets one message from each topic, thanks to the approximate
    # time synchronizer.
    def thruster_callback(self, t0, t1, t2, t3, t4, t5, t6, t7):
        # TODO: find a good way to support arbitrary numbers of thruster commands
        messages = [t0, t1, t2, t3, t4, t5, t6, t7]

        for i, m in enumerate(messages):
            val = m.data
            self.thruster_command[self.thrusters[i]] = val

        # print("here: {}".format(rospy.get_rostime()))
        self.last_thruster_msg = rospy.get_rostime()

    # goes from normalized (-1 to 1) speed to uS pulse width
    def speed_to_pwm(self, cmd, channel):
        if cmd < -1:
            print("Thruster Command out of range!")
            cmd = -1
        if cmd > 1:
            print("Thruster Command out of range!")
            cmd = 1
        halfrange = (self.pwm_forward - self.pwm_reverse)/2.0
        zero = halfrange + self.pwm_reverse
        return int(round(zero + halfrange * cmd * self.thruster_inversions[channel]))

    def srv_arm(self, req):
        self.armed = req.data
        return SetBoolResponse(True, "")


def main():
    rospy.init_node('actuator_controller')
    a = ActuatorController()
    a.start()

