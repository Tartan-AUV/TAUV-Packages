import rospy
import argparse
from typing import Optional
import numpy as np
from math import atan2, cos, sin, e, pi

from tauv_msgs.msg import PIDTuning, DynamicsTuning, DynamicsParameterConfigUpdate
from tauv_msgs.srv import \
    TuneController, TuneControllerRequest,\
    TunePIDPlanner, TunePIDPlannerRequest,\
    TuneDynamics, TuneDynamicsRequest,\
    UpdateDynamicsParameterConfigs, UpdateDynamicsParameterConfigsRequest, UpdateDynamicsParameterConfigsResponse
from geometry_msgs.msg import Pose, Twist, Point
from std_srvs.srv import SetBool, Trigger
from std_msgs.msg import Float64
from tauv_msgs.srv import MapFind, MapFindRequest, MapFindClosest, MapFindClosestRequest
from motion.motion_utils import MotionUtils
from motion.trajectories import TrajectoryStatus


class ArgumentParserError(Exception): pass


class ThrowingArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        raise ArgumentParserError(message)


class TeleopMission:

    def __init__(self):
        self._parser = self._build_parser()

        self._motion = MotionUtils()

        self._pose: Optional[Pose] = None
        self._twist: Optional[Twist] = None

        self._tune_controller_srv: rospy.ServiceProxy = rospy.ServiceProxy('gnc/tune_controller', TuneController)
        self._tune_pid_planner_srv: rospy.ServiceProxy = rospy.ServiceProxy('gnc/tune_pid_planner', TunePIDPlanner)
        self._tune_dynamics_srv: rospy.ServiceProxy = rospy.ServiceProxy('gnc/tune_dynamics', TuneDynamics)
        self._arm_srv: rospy.ServiceProxy = rospy.ServiceProxy('vehicle/thrusters/arm', SetBool)
        self._update_dynamics_parameter_configs_srv: rospy.ServiceProxy = rospy.ServiceProxy(
            'gnc/update_dynamics_parameter_configs', UpdateDynamicsParameterConfigs
        )

        self._torpedo_servo_pub: rospy.Publisher = rospy.Publisher('vehicle/servos/0/target_position', Float64, queue_size=10)
        self._arm_servo_pub: rospy.Publisher = rospy.Publisher('vehicle/servos/4/target_position', Float64, queue_size=10)
        self._suction_servo_pub: rospy.Publisher = rospy.Publisher('vehicle/servos/5/target_position', Float64, queue_size=10)

        self._goto_circle_timer: rospy.Timer = None
        self._last_circle_position = None
        self._goto_circle_args = None
        self._goto_circle_v = 0.1
        self._goto_circle_a = 0.1
        self._goto_circle_j = 0.4
        self._goto_circle_state = None

        self._find_srv = rospy.ServiceProxy("global_map/find", MapFind)
        self._find_closest_srv = rospy.ServiceProxy("global_map/find_closest", MapFindClosest)
        self._map_reset_srv = rospy.ServiceProxy("global_map/reset", Trigger)

        self._pick_chevron_args = None
        self._pick_chevron_timer = None
        self._last_chevron_position = None

        self._prequal_timer = None

    def start(self):
        while not rospy.is_shutdown():
            cmd = input('>>> ')
            try:
                args = self._parser.parse_args(cmd.split())
                args.func(args)
            except ArgumentParserError as e:
                print('error:', e)
                continue

    def _handle_tune_controller(self, args):
        print('tune_controller', args.z, args.roll, args.pitch)
        pid_tunings = []

        if args.z is not None:
            p = PIDTuning(
                axis="z",
                kp=args.z[0],
                ki=args.z[1],
                kd=args.z[2],
                tau=args.z[3],
                limits=args.z[4:6]
            )
            pid_tunings.append(p)

        if args.roll is not None:
            p = PIDTuning(
                axis="roll",
                kp=args.roll[0],
                ki=args.roll[1],
                kd=args.roll[2],
                tau=args.roll[3],
                limits=args.roll[4:6]
            )
            pid_tunings.append(p)

        if args.pitch is not None:
            p = PIDTuning(
                axis="pitch",
                kp=args.pitch[0],
                ki=args.pitch[1],
                kd=args.pitch[2],
                tau=args.pitch[3],
                limits=args.pitch[4:6]
            )
            pid_tunings.append(p)

        req = TuneControllerRequest()
        req.tunings = pid_tunings
        try:
            self._tune_controller_srv.call(req)
        except Exception as e:
            print(e)

    def _handle_tune_pid_planner(self, args):
        print('tune_pid_planner', args.x, args.y, args.z, args.roll, args.pitch, args.yaw)

        pid_tunings = []

        if args.x is not None:
            p = PIDTuning(
                axis="x",
                kp=args.x[0],
                ki=args.x[1],
                kd=args.x[2],
                tau=args.x[3],
                limits=args.x[4:6]
            )
            pid_tunings.append(p)

        if args.y is not None:
            p = PIDTuning(
                axis="y",
                kp=args.y[0],
                ki=args.y[1],
                kd=args.y[2],
                tau=args.y[3],
                limits=args.y[4:6]
            )
            pid_tunings.append(p)

        if args.z is not None:
            p = PIDTuning(
                axis="z",
                kp=args.z[0],
                ki=args.z[1],
                kd=args.z[2],
                tau=args.z[3],
                limits=args.z[4:6]
            )
            pid_tunings.append(p)

        if args.roll is not None:
            p = PIDTuning(
                axis="roll",
                kp=args.roll[0],
                ki=args.roll[1],
                kd=args.roll[2],
                tau=args.roll[3],
                limits=args.roll[4:6]
            )
            pid_tunings.append(p)

        if args.pitch is not None:
            p = PIDTuning(
                axis="pitch",
                kp=args.pitch[0],
                ki=args.pitch[1],
                kd=args.pitch[2],
                tau=args.pitch[3],
                limits=args.pitch[4:6]
            )
            pid_tunings.append(p)

        if args.yaw is not None:
            p = PIDTuning(
                axis="yaw",
                kp=args.yaw[0],
                ki=args.yaw[1],
                kd=args.yaw[2],
                tau=args.yaw[3],
                limits=[args.yaw[4], args.yaw[5]]
            )
            pid_tunings.append(p)

        req = TunePIDPlannerRequest()
        req.tunings = pid_tunings
        try:
            self._tune_pid_planner_srv.call(req)
        except Exception as e:
            print(e)

    def _handle_tune_dynamics(self, args):
        print('tune_dynamics', args.mass, args.volume, args.water_density, args.center_of_gravity,
              args.center_of_buoyancy, args.moments, args.linear_damping, args.quadratic_damping, args.added_mass)

        t = DynamicsTuning()
        if args.mass is not None:
            t.update_mass = True
            t.mass = args.mass

        if args.volume is not None:
            t.update_volume = True
            t.volume = args.volume

        if args.water_density is not None:
            t.update_water_density = True
            t.water_density = args.water_density

        if args.center_of_gravity is not None:
            t.update_center_of_gravity = True
            t.center_of_gravity = args.center_of_gravity

        if args.center_of_buoyancy is not None:
            t.update_center_of_buoyancy = True
            t.center_of_buoyancy = args.center_of_buoyancy

        if args.moments is not None:
            t.update_moments = True
            t.moments = args.moments

        if args.linear_damping is not None:
            t.update_linear_damping = True
            t.linear_damping = args.linear_damping

        if args.quadratic_damping is not None:
            t.update_quadratic_damping = True
            t.quadratic_damping = args.quadratic_damping

        if args.added_mass is not None:
            t.update_added_mass = True
            t.added_mass = args.added_mass

        req: TuneDynamicsRequest = TuneDynamicsRequest()
        req.tuning = t
        self._tune_dynamics_srv.call(req)

    def _handle_goto(self, args):
        v = args.v if args.v is not None else .1
        a = args.a if args.a is not None else .1
        j = args.j if args.j is not None else .4

        try:
            self._motion.goto(
                (args.x, args.y, args.z),
                args.yaw,
                v=v,
                a=a,
                j=j,
                block=TrajectoryStatus.EXECUTING
            )
        except Exception as e:
            print("Exception from teleop_mission! (Gleb)")
            print(e)

    def _handle_goto_relative(self, args):
        v = args.v if args.v is not None else .1
        a = args.a if args.a is not None else .1
        j = args.j if args.j is not None else .4

        try:
            self._motion.goto_relative(
                (args.x, args.y, args.z),
                args.yaw,
                v=v,
                a=a,
                j=j
            )
        except Exception as e:
            print(e)

    def _handle_goto_circle(self, args):
        v = args.v if args.v is not None else .1
        a = args.a if args.a is not None else .1
        j = args.j if args.j is not None else .4

        self._goto_circle_v = v
        self._goto_circle_a = a
        self._goto_circle_j = j

        self._goto_circle_args = args

        self._last_circle_position = None

        if self._goto_circle_timer is not None:
            self._goto_circle_timer.shutdown()

        self._goto_circle_state = 0

        self._motion.reset()

        self._goto_circle_timer = rospy.Timer(rospy.Duration(0.1), self._update_goto_circle)

    def _handle_stop_goto_circle(self, args):
        if self._goto_circle_timer is not None:
            self._goto_circle_timer.shutdown()

    def _update_goto_circle(self, timer_event):
        args = self._goto_circle_args

        sub_position = self._motion.get_position()
        sub_yaw = self._motion.get_orientation()[2]

        req = MapFindClosestRequest()
        req.tag = 'circle'

        if self._last_circle_position is None:
            req.point = Point(sub_position[0], sub_position[1], sub_position[2])
        else:
            req.point = Point(self._last_circle_position[0], self._last_circle_position[1], self._last_circle_position[2])
        # detections, success = self._find_srv.call()
        resp = self._find_closest_srv.call(req)

        if not resp.success:
            print('no circle')
            return

        detection = resp.detection

        # Use detection
        circle_position = np.array([detection.position.x, detection.position.y, detection.position.z])
        circle_yaw = detection.orientation.z

        approach_offset = -np.array([
            args.ax, args.ay, args.az
        ])

        shoot_offset = -np.array([
            args.bx, args.by, args.bz
        ])

        approach_yaw = atan2(circle_position[1] - sub_position[1], circle_position[0] - sub_position[0])

        approach_position = circle_position + np.array([
            approach_offset[0] * cos(approach_yaw) + approach_offset[1] * -sin(approach_yaw),
            approach_offset[0] * sin(approach_yaw) + approach_offset[1] * cos(approach_yaw),
            approach_offset[2]
        ])

        shoot_position = circle_position + np.array([
            shoot_offset[0] * cos(circle_yaw) + shoot_offset[1] * -sin(circle_yaw),
            shoot_offset[0] * sin(circle_yaw) + shoot_offset[1] * cos(circle_yaw),
            shoot_offset[2]
        ])

        shoot_error = shoot_position - sub_position

        circle_direction = np.array([
            cos(approach_yaw),
            sin(approach_yaw),
            0
        ])

        planar_error = shoot_error - np.dot(shoot_error, circle_direction) * circle_direction

        shoot_decay_position = sub_position + planar_error + (e ** (-args.w * np.linalg.norm(planar_error))) * np.dot(shoot_error, circle_direction) * circle_direction

        if self._goto_circle_state == 0 and np.linalg.norm(sub_position - approach_position) < 0.05 and np.abs(sub_yaw - approach_yaw) < 0.05:
            print('lined up')
            self._goto_circle_state = 1

        if self._goto_circle_state == 1 and np.linalg.norm(sub_position - shoot_position) < 0.05 and np.abs(sub_yaw - approach_yaw) < 0.05:
            print('shoot!')
            # rospy.sleep(1.0)
            # self._motion.shoot_torpedo(0)
            # rospy.sleep(1)
            self._motion.shoot_torpedo(1)

        if self._goto_circle_state == 0:
            self._motion.goto(approach_position, approach_yaw, v=self._goto_circle_v, a=self._goto_circle_a, j=self._goto_circle_j, block=TrajectoryStatus.EXECUTING)

        if self._goto_circle_state == 1:
            self._motion.goto(shoot_decay_position, approach_yaw, v=self._goto_circle_v, a=self._goto_circle_a, j=self._goto_circle_j, block=TrajectoryStatus.EXECUTING)

    def _handle_pick_chevron(self, args):
        v = args.v if args.v is not None else .1
        a = args.a if args.a is not None else .1
        j = args.j if args.j is not None else .4

        self._map_reset_srv.call()

        self._pick_chevron_v = v
        self._pick_chevron_a = a
        self._pick_chevron_j = j
        self._pick_chevron_args = args

        self._motion.reset()

        if self._pick_chevron_timer is not None:
            self._pick_chevron_timer.shutdown()

        self._last_chevron_position = None

        self._pick_chevron_timer = rospy.Timer(rospy.Duration(args.r), self._handle_update_pick_chevron)

    def _handle_update_pick_chevron(self, timer_event):
        args = self._pick_chevron_args

        sub_position = self._motion.get_position()

        req = MapFindClosestRequest()
        req.tag = 'chevron'
        if self._last_chevron_position is None:
            req.point = Point(sub_position[0], sub_position[1], sub_position[2])
        else:
            req.point = Point(self._last_chevron_position[0], self._last_chevron_position[1], self._last_chevron_position[2])
        resp = self._find_closest_srv.call(req)

        if not resp.success:
            print('no chevron')
            return

        detection = resp.detection

        approach_yaw = detection.orientation.z

        chevron_position = np.array([detection.position.x, detection.position.y, detection.position.z])
        self._last_chevron_position = chevron_position
        suction_position = np.array([0.1524, -0.0508, 0.4572])
        suction_offset = -(np.array([args.x, args.y, args.z]) + suction_position)
        goal_position = chevron_position + np.array([
            suction_offset[0] * cos(approach_yaw) + suction_offset[1] * -sin(approach_yaw),
            suction_offset[0] * sin(approach_yaw) + suction_offset[1] * cos(approach_yaw),
            suction_offset[2]
        ])

        sub_orientation = self._motion.get_orientation()

        error = args.wxy * np.linalg.norm(goal_position[0:2] - sub_position[0:2]) + args.wy * np.abs(sub_orientation[2] - approach_yaw)

        target_position = goal_position + (1 - (e ** (-error))) * np.array([0, 0, sub_position[2] - goal_position[2]])

        if np.linalg.norm(sub_position - goal_position) < args.t:
            self._pick_chevron_timer.shutdown()
            self._suction_servo_pub.publish(-90)
            self._motion.goto(
                (goal_position[0], goal_position[1], goal_position[2] + args.s),
                approach_yaw,
                v=self._pick_chevron_v,
                a=self._pick_chevron_a,
                j=self._pick_chevron_j,
                block=TrajectoryStatus.FINISHED
            )
            self._motion.goto(
                (target_position[0], target_position[1], 0),
                approach_yaw,
                v=self._pick_chevron_v,
                a=self._pick_chevron_a,
                j=self._pick_chevron_j,
                block=TrajectoryStatus.FINISHED
            )
        else:
            self._motion.goto(
                (target_position[0], target_position[1], target_position[2]),
                approach_yaw,
                v=self._pick_chevron_v,
                a=self._pick_chevron_a,
                j=self._pick_chevron_j,
                block=TrajectoryStatus.EXECUTING
            )

    def _handle_stop_pick_chevron(self, args):
        if self._pick_chevron_timer is not None:
            self._pick_chevron_timer.shutdown()

    def _handle_shoot_torpedo(self, args):
        print(f'shoot torpedo {args.torpedo}')

        self._motion.shoot_torpedo(args.torpedo)

        # if args.torpedo == 0:
        #     self._torpedo_servo_pub.publish(90)
        #     rospy.sleep(1.0)
        #     self._torpedo_servo_pub.publish(0)
        # elif args.torpedo == 1:
        #     self._torpedo_servo_pub.publish(-90)
        #     rospy.sleep(1.0)
        #     self._torpedo_servo_pub.publish(0)
        # else:
        #     self._torpedo_servo_pub.publish(0)

    def _handle_drop_marker(self, args):
        print(f'drop marker {args.marker}')

        self._motion.drop_marker(args.marker)

    def _handle_move_arm(self, args):
        print(f'move arm {args.position}')

        self._arm_servo_pub.publish(args.position)

    def _handle_suction(self, args):
        print(f'suction {args.power}')

        self._suction_servo_pub.publish(args.power * (90 / 100))

    def _handle_arm(self, args):
        print('arm')

        self._arm_srv.call(True)

    def _handle_disarm(self, args):
        print('disarm')

        self._arm_srv.call(False)

    def _handle_config_param_est(self, args):
        req = UpdateDynamicsParameterConfigsRequest()
        update = DynamicsParameterConfigUpdate()
        update.name = args.name

        if args.initial_value is not None:
            update.update_initial_value = True
            update.initial_value = args.initial_value

        if args.fixed is not None:
            update.update_fixed = True
            update.fixed = args.fixed == "true"

        if args.initial_covariance is not None:
            update.update_initial_covariance = True
            update.initial_covariance = args.initial_covariance

        if args.process_covariance is not None:
            update.update_process_covariance = True
            update.process_covariance = args.process_covariance

        if args.limits is not None:
            update.update_limits = True
            update.limits = args.limits

        update.reset = args.reset

        req.updates = [update]

        self._update_dynamics_parameter_configs_srv.call(req)

    def _handle_prequal(self, args):
        if self._prequal_timer is not None:
            self._prequal_timer.shutdown()

        self._prequal_timer = rospy.Timer(rospy.Duration(30), self._handle_update_prequal, oneshot=True)

    def _handle_update_prequal(self, timer_event):
        self._motion.arm(True)
        self._motion.reset()

        print('running!')

        depth = 1.5

        start_position = self._motion.get_position()
        start_yaw = self._motion.get_orientation()[2]

        waypoints = np.array([
            [0, 0, depth, 0],
            [3, 0, depth, 0],
            [12, 2, depth, 0],
            [12, -2, depth, 0],
            [3, 0, depth, 0],
            [0, 0, depth, 0],
        ])
        n_waypoints = waypoints.shape[0]

        for i in range(n_waypoints):
            position = waypoints[i, 0:3]
            yaw = waypoints[i, 3]

            transformed_position = start_position + np.array([
                position[0] * cos(start_yaw) + position[1] * -sin(start_yaw),
                position[0] * sin(start_yaw) + position[1] * cos(start_yaw),
                position[2]
            ])

            transformed_yaw = start_yaw + yaw

            self._motion.goto(transformed_position, transformed_yaw, v=0.3, a=0.05, j=0.04)

        self._motion.arm(False)

    def _build_parser(self) -> argparse.ArgumentParser:
        parser = ThrowingArgumentParser(prog="teleop_mission")
        subparsers = parser.add_subparsers()

        tune_controller = subparsers.add_parser('tune_controller')
        tune_controller.add_argument('--z', type=float, nargs=6)
        tune_controller.add_argument('--roll', type=float, nargs=6)
        tune_controller.add_argument('--pitch', type=float, nargs=6)
        tune_controller.set_defaults(func=self._handle_tune_controller)

        tune_pid_planner = subparsers.add_parser('tune_pid_planner')
        tune_pid_planner.add_argument('--x', type=float, nargs=6)
        tune_pid_planner.add_argument('--y', type=float, nargs=6)
        tune_pid_planner.add_argument('--z', type=float, nargs=6)
        tune_pid_planner.add_argument('--roll', type=float, nargs=6)
        tune_pid_planner.add_argument('--pitch', type=float, nargs=6)
        tune_pid_planner.add_argument('--yaw', type=float, nargs=6)
        tune_pid_planner.set_defaults(func=self._handle_tune_pid_planner)

        tune_dynamics = subparsers.add_parser('tune_dynamics')
        tune_dynamics.add_argument('--mass', type=float)
        tune_dynamics.add_argument('--volume', type=float)
        tune_dynamics.add_argument('--water_density', type=float)
        tune_dynamics.add_argument('--center_of_gravity', type=float, nargs=3)
        tune_dynamics.add_argument('--center_of_buoyancy', type=float, nargs=3)
        tune_dynamics.add_argument('--moments', type=float, nargs=6)
        tune_dynamics.add_argument('--linear_damping', type=float, nargs=6)
        tune_dynamics.add_argument('--quadratic_damping', type=float, nargs=6)
        tune_dynamics.add_argument('--added_mass', type=float, nargs=6)
        tune_dynamics.set_defaults(func=self._handle_tune_dynamics)

        goto = subparsers.add_parser('goto')
        goto.add_argument('x', type=float)
        goto.add_argument('y', type=float)
        goto.add_argument('z', type=float)
        goto.add_argument('yaw', type=float)
        goto.add_argument('--v', type=float)
        goto.add_argument('--a', type=float)
        goto.add_argument('--j', type=float)
        goto.set_defaults(func=self._handle_goto)

        goto_relative = subparsers.add_parser('goto_relative')
        goto_relative.add_argument('x', type=float)
        goto_relative.add_argument('y', type=float)
        goto_relative.add_argument('z', type=float)
        goto_relative.add_argument('yaw', type=float)
        goto_relative.add_argument('--v', type=float)
        goto_relative.add_argument('--a', type=float)
        goto_relative.add_argument('--j', type=float)
        goto_relative.set_defaults(func=self._handle_goto_relative)

        goto_circle = subparsers.add_parser('goto_circle')
        goto_circle.add_argument('--v', type=float)
        goto_circle.add_argument('--a', type=float)
        goto_circle.add_argument('--j', type=float)
        goto_circle.add_argument('--ax', type=float)
        goto_circle.add_argument('--ay', type=float)
        goto_circle.add_argument('--az', type=float)
        goto_circle.add_argument('--bx', type=float)
        goto_circle.add_argument('--by', type=float)
        goto_circle.add_argument('--bz', type=float)
        goto_circle.add_argument('--w', type=float)
        goto_circle.set_defaults(func=self._handle_goto_circle)

        stop_goto_circle = subparsers.add_parser('stop_goto_circle')
        stop_goto_circle.set_defaults(func=self._handle_stop_goto_circle)

        pick_chevron = subparsers.add_parser('pick_chevron')
        pick_chevron.add_argument('--v', type=float)
        pick_chevron.add_argument('--a', type=float)
        pick_chevron.add_argument('--j', type=float)
        pick_chevron.add_argument('--x', type=float, default=0)
        pick_chevron.add_argument('--y', type=float, default=0)
        pick_chevron.add_argument('--z', type=float, default=0)
        pick_chevron.add_argument('--s', type=float, default=0)
        pick_chevron.add_argument('--wxy', type=float, default=10)
        pick_chevron.add_argument('--wy', type=float, default=10)
        pick_chevron.add_argument('--r', type=float, default=1.0)
        pick_chevron.add_argument('--t', type=float, default=0.05)
        pick_chevron.set_defaults(func=self._handle_pick_chevron)

        stop_pick_chevron = subparsers.add_parser('stop_pick_chevron')
        stop_pick_chevron.set_defaults(func=self._handle_stop_pick_chevron)

        shoot_torpedo = subparsers.add_parser('shoot_torpedo')
        shoot_torpedo.add_argument('torpedo', type=int)
        shoot_torpedo.set_defaults(func=self._handle_shoot_torpedo)

        drop_marker = subparsers.add_parser('drop_marker')
        drop_marker.add_argument('marker', type=int)
        drop_marker.set_defaults(func=self._handle_drop_marker)

        move_arm = subparsers.add_parser('move_arm')
        move_arm.add_argument('position', type=float)
        move_arm.set_defaults(func=self._handle_move_arm)

        suction = subparsers.add_parser('suction')
        suction.add_argument('power', type=float)
        suction.set_defaults(func=self._handle_suction)

        arm = subparsers.add_parser('arm')
        arm.set_defaults(func=self._handle_arm)

        disarm = subparsers.add_parser('disarm')
        disarm.set_defaults(func=self._handle_disarm)

        config_param_est = subparsers.add_parser('config_param_est')
        config_param_est.add_argument('name', type=str)
        config_param_est.add_argument('--initial_value', type=float)
        config_param_est.add_argument('--fixed', type=str, choices=('true', 'false'))
        config_param_est.add_argument('--initial_covariance', type=float)
        config_param_est.add_argument('--process_covariance', type=float)
        config_param_est.add_argument('--limits', type=float, nargs=2)
        config_param_est.add_argument('--reset', default=False, action='store_true')
        config_param_est.set_defaults(func=self._handle_config_param_est)

        prequal = subparsers.add_parser('prequal')
        prequal.set_defaults(func=self._handle_prequal)

        return parser


def main():
    rospy.init_node('teleop_mission')
    m = TeleopMission()
    m.start()