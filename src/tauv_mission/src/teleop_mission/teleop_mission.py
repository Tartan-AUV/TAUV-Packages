import rospy
import argparse
from typing import Optional
from threading import Thread
from motion_client import MotionClient
from actuator_client import ActuatorClient
from map_client import MapClient
from transform_client import TransformClient
from tauv_msgs.msg import PIDTuning, DynamicsTuning, DynamicsParameterConfigUpdate
from tauv_msgs.srv import \
    TuneController, TuneControllerRequest,\
    TunePIDPlanner, TunePIDPlannerRequest,\
    TuneDynamics, TuneDynamicsRequest,\
    UpdateDynamicsParameterConfigs, UpdateDynamicsParameterConfigsRequest, UpdateDynamicsParameterConfigsResponse
from std_srvs.srv import SetBool
from spatialmath import SE3, SO3, SE2, SO2
import numpy as np
from tasks import Task, TaskResources
from tasks.pick_chevron import PickChevron as PickChevronTask, PickChevronResult, PickChevronStatus
from tasks.dive import Dive as DiveTask, DiveResult, DiveStatus
from tasks.scan_rotate import ScanRotate as ScanRotateTask
from tasks.scan_translate import ScanTranslate as ScanTranslateTask
from tasks.hit_buoy import HitBuoy as HitBuoyTask
from tasks.detect_pinger import DetectPinger as DetectPingerTask
from tasks.barrel_roll import BarrelRoll as BarrelRollTask
from tasks.gate import Gate as GateTask
from tasks.shoot_torpedo import ShootTorpedo as ShootTorpedoTask
from tasks.torpedo import Torpedo as TorpedoTask
from tasks.collect_sample import CollectSample as CollectSampleTask
from tasks.torpedo_24 import Torpedo24 as Torpedo24Task
from tasks.buoy_24 import CircleBuoy as CircleBuoyTask
from tasks.debug_depth_task import DebugDepth as DebugDepthTask
from tasks.approach_samples import  ApproachSamples as ApproachSamplesTask

class ArgumentParserError(Exception): pass


class ThrowingArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        raise ArgumentParserError(message)

class TeleopMission:

    def __init__(self):
        self._parser = self._build_parser()

        self._motion = MotionClient()
        self._actuators = ActuatorClient()
        self._map = MapClient()
        self._transforms = TransformClient()

        self._task_resources: TaskResources = TaskResources(self._motion, self._actuators, self._map, self._transforms)
        self._task: Optional[Task] = None

        self._tune_controller_srv: rospy.ServiceProxy = rospy.ServiceProxy('gnc/tune_controller', TuneController)
        self._tune_pid_planner_srv: rospy.ServiceProxy = rospy.ServiceProxy('gnc/tune_pid_planner', TunePIDPlanner)
        self._tune_dynamics_srv: rospy.ServiceProxy = rospy.ServiceProxy('gnc/tune_dynamics', TuneDynamics)
        self._arm_srv: rospy.ServiceProxy = rospy.ServiceProxy('vehicle/thrusters/arm', SetBool)
        self._update_dynamics_parameter_configs_srv: rospy.ServiceProxy = rospy.ServiceProxy(
            'gnc/update_dynamics_parameter_configs', UpdateDynamicsParameterConfigs
        )

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

    def _handle_arm(self, args):
        print('arm')

        self._arm_srv.call(True)

    def _handle_disarm(self, args):
        print('disarm')

        self._arm_srv.call(False)

    def _handle_goto(self, args):
        print('goto')
        pose = SE3.Rt(SO3.Rx(np.deg2rad(args.yaw)), np.array([args.x, args.y, args.z]))

        self._motion.goto(pose)

    def _handle_goto_relative(self, args):
        print('goto')

        if args.relative_z:
            self._motion.goto_relative(
                SE3.Rt(SO3.Rx(args.yaw), np.array([args.x, args.y, args.z]))
            )
        else:
            self._motion.goto_relative_with_depth(
                SE2(args.x, args.y, np.deg2rad(args.yaw)),
                args.z
            )

    def _handle_shoot_torpedo(self, args):
        print('shoot_torpedo')

        self._actuators.shoot_torpedo(args.torpedo)

    def _handle_drop_marker(self, args):
        print('drop_marker')

        self._actuators.drop_marker(args.marker)

    def _handle_move_arm(self, args):
        print('move_arm')

        self._actuators.move_arm(args.position)

    def _handle_activate_suction(self, args):
        print('activate_suction')

        self._actuators.activate_suction(args.strength)

    def _handle_run_shoot_torpedo_task(self, args):
        print('run_shoot_torpedo_task')

        if self._task is not None:
            print('task in progress')
            return

        self._task = ShootTorpedoTask()
        Thread(target=self._run_task, daemon=True).start()

    def _handle_run_pick_chevron_task(self, args):
        print('run_pick_chevron_task')

        if self._task is not None:
            print('task in progress')
            return

        self._task = PickChevronTask()
        Thread(target=self._run_task, daemon=True).start()

    def _handle_run_dive_task(self, args):
        print('run_dive_task')

        if self._task is not None:
            print('task in progress')
            return

        self._task = DiveTask(args.delay)
        Thread(target=self._run_task, daemon=True).start()

    def _handle_run_scan_rotate_task(self, args):
        print('run_scan_rotate_task')

        if self._task is not None:
            print('task in progress')
            return

        self._task = ScanRotateTask()
        Thread(target=self._run_task, daemon=True).start()

    def _handle_run_scan_translate_task(self, args):
        print('run_scan_translate_task')

        if self._task is not None:
            print('task in progress')
            return

        self._task = ScanTranslateTask()
        Thread(target=self._run_task, daemon=True).start()

    def _handle_run_hit_buoy_task(self, args):
        print('run_hit_buoy_task')

        if self._task is not None:
            print('task in progress')
            return

        self._task = HitBuoyTask(args.tag, args.timeout, args.frequency, args.distance, args.error_a, args.error_b, args.error_threshold, args.shoot_torpedo)
        Thread(target=self._run_task, daemon=True).start()

    def _handle_run_collect_sample_task(self, args):
        print('run_collect_sample_task')

        if self._task is not None:
            print('task in progress')
            return

        self._task = CollectSampleTask(args.tag, args.timeout, args.frequency, args.distance, args.error_a, args.error_b, args.error_threshold)
        Thread(target=self._run_task, daemon=True).start()

    def _handle_run_torpedo_24_task(self, args):
        print('run_torpedo_24_task')

        if self._task is not None:
            print('task in progress')
            return

        self._task = Torpedo24Task(args.timeout, args.frequency, False)
        Thread(target=self._run_task, daemon=True).start()

    def _handle_run_torpedo_task(self, args):
        print('run_torpedo_task')

        if self._task is not None:
            print('task in progress')
            return

        if args.torpedo == -1:
            args.torpedo = None

        self._task = TorpedoTask(
            'torpedo_22_trapezoid',
            1000,
            10,
            10,
            1,
            0.5,
            0.05,
            0.05,
            0.1,
            0.05,
            args.torpedo
        )
        Thread(target=self._run_task, daemon=True).start()

    def _handle_run_shoot_torpedo_task(self, args):
        if self._task is not None:
            return

        if args.torpedo == -1:
            args.torpedo = None

        self._task = ShootTorpedoTask(args.tag, args.torpedo, args.timeout, args.frequency, args.distance, args.error_factor, args.error_threshold)
        Thread(target=self._run_task, daemon=True).start()

    def _handle_run_detect_pinger_task(self, args):
        if self._task is not None:
            print('task in progress')
            return

        self._task = DetectPingerTask(args.frequency, args.depth)
        Thread(target=self._run_task, daemon=True).start()

    def _handle_run_gate_task(self, args):
        self._task = GateTask()
        Thread(target=self._run_task, daemon=True).start()
    
    def _handle_circle_buoy(self, args):
        self._task = CircleBuoyTask("buoy_24", args.radius, circle_ccw=True, waypoint_every_n_meters=0.3,
                                    latch_buoy=args.latch)
        Thread(target=self._run_task, daemon=True).start()

    def _handle_debug_depth(self, args):
        self._task = DebugDepthTask()
        Thread(target=self._run_task, daemon=True).start()

    def _handle_approach_samples(self, args):
        self._task = ApproachSamplesTask(["sample_24_coral", "sample_24_nautilus"])
        Thread(target=self._run_task, daemon=True).start()

    def _run_task(self):
        self._task.run(self._task_resources)
        self._task = None

    def _handle_cancel_task(self, args):
        print('cancel_task')

        if self._task is not None:
            self._task.cancel()
            self._task = None

    def _handle_cancel(self, args):
        self._motion.cancel()

    def _handle_set_eater(self, args):
        print(f'set_eater {args.direction}')

        self._actuators.set_eater(args.direction)

    def _handle_set_sphincter(self, args):
        print(f'set_sphincter {args.open} {args.strength} {args.duration}')

        self._actuators.set_sphincter(args.open, args.strength, args.duration)

    def _handle_open_sphincter(self, args):
        self._actuators.open_sphincter()

    def _handle_barrel_roll(self, args):
        self._task = BarrelRollTask()
        Thread(target=self._run_task, daemon=True).start()

    def _handle_close_sphincter(self, args):
        self._actuators.close_sphincter()

    def _handle_clear_map(self, args):
        self._map.reset()

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
        goto_relative.add_argument('--relative-z', default=False, action='store_true')
        goto_relative.add_argument('--v', type=float)
        goto_relative.add_argument('--a', type=float)
        goto_relative.add_argument('--j', type=float)
        goto_relative.set_defaults(func=self._handle_goto_relative)

        config_param_est = subparsers.add_parser('config_param_est')
        config_param_est.add_argument('name', type=str)
        config_param_est.add_argument('--initial_value', type=float)
        config_param_est.add_argument('--fixed', type=str, choices=('true', 'false'))
        config_param_est.add_argument('--initial_covariance', type=float)
        config_param_est.add_argument('--process_covariance', type=float)
        config_param_est.add_argument('--limits', type=float, nargs=2)
        config_param_est.add_argument('--reset', default=False, action='store_true')
        config_param_est.set_defaults(func=self._handle_config_param_est)

        arm = subparsers.add_parser('arm')
        arm.set_defaults(func=self._handle_arm)

        disarm = subparsers.add_parser('disarm')
        disarm.set_defaults(func=self._handle_disarm)

        cancel = subparsers.add_parser('cancel')
        cancel.set_defaults(func=self._handle_cancel)

        shoot_torpedo = subparsers.add_parser('shoot_torpedo')
        shoot_torpedo.add_argument('torpedo', type=int)
        shoot_torpedo.set_defaults(func=self._handle_shoot_torpedo)

        drop_marker = subparsers.add_parser('drop_marker')
        drop_marker.add_argument('marker', type=int)
        drop_marker.set_defaults(func=self._handle_drop_marker)

        move_arm = subparsers.add_parser('move_arm')
        move_arm.add_argument('position', type=float)
        move_arm.set_defaults(func=self._handle_move_arm)

        activate_suction = subparsers.add_parser('set_suction')
        activate_suction.add_argument('strength', type=float)
        activate_suction.set_defaults(func=self._handle_activate_suction)

        run_shoot_torpedo_task = subparsers.add_parser('run_shoot_torpedo_task')
        run_shoot_torpedo_task.set_defaults(func=self._handle_run_shoot_torpedo_task)

        run_pick_chevron_task = subparsers.add_parser('run_pick_chevron_task')
        run_pick_chevron_task.set_defaults(func=self._handle_run_pick_chevron_task)

        run_dive_task = subparsers.add_parser('run_dive_task')
        run_dive_task.add_argument("delay", type=float)
        run_dive_task.set_defaults(func=self._handle_run_dive_task)

        run_scan_rotate_task = subparsers.add_parser('run_scan_rotate_task')
        run_scan_rotate_task.set_defaults(func=self._handle_run_scan_rotate_task)

        run_scan_translate_task = subparsers.add_parser('run_scan_translate_task')
        run_scan_translate_task.add_argument('x_range', type=float)
        run_scan_translate_task.add_argument('y_range', type=float)
        run_scan_translate_task.set_defaults(func=self._handle_run_scan_translate_task)

        run_hit_buoy_task = subparsers.add_parser('run_hit_buoy_task')
        run_hit_buoy_task.add_argument('tag', type=str)
        run_hit_buoy_task.add_argument('timeout', type=float)
        run_hit_buoy_task.add_argument('frequency', type=float)
        run_hit_buoy_task.add_argument('distance', type=float)
        run_hit_buoy_task.add_argument('error_a', type=float)
        run_hit_buoy_task.add_argument('error_b', type=float)
        run_hit_buoy_task.add_argument('error_threshold', type=float)
        run_hit_buoy_task.add_argument('shoot_torpedo', type=int)
        run_hit_buoy_task.set_defaults(func=self._handle_run_hit_buoy_task)

        barrel_roll = subparsers.add_parser('barrel_roll')
        barrel_roll.set_defaults(func=self._handle_barrel_roll)

        run_collect_sample_task = subparsers.add_parser('run_collect_sample_task')
        run_collect_sample_task.add_argument('tag', type=str)
        run_collect_sample_task.add_argument('timeout', type=float)
        run_collect_sample_task.add_argument('frequency', type=float)
        run_collect_sample_task.add_argument('distance', type=float)
        run_collect_sample_task.add_argument('error_a', type=float)
        run_collect_sample_task.add_argument('error_b', type=float)
        run_collect_sample_task.add_argument('error_threshold', type=float)
        run_collect_sample_task.set_defaults(func=self._handle_run_collect_sample_task)

        run_torpedo_24_task = subparsers.add_parser('run_torpedo_24_task')
        run_torpedo_24_task.add_argument('timeout', type=float)
        run_torpedo_24_task.add_argument('frequency', type=float)
        run_torpedo_24_task.set_defaults(func=self._handle_run_torpedo_24_task)


        run_torpedo_task = subparsers.add_parser('run_torpedo_task')
        run_torpedo_task.add_argument('torpedo', type=int)
        run_torpedo_task.set_defaults(func=self._handle_run_torpedo_task)

        run_shoot_torpedo_task = subparsers.add_parser('run_shoot_torpedo_task')
        run_shoot_torpedo_task.add_argument('tag', type=str)
        run_shoot_torpedo_task.add_argument('torpedo', type=int)
        run_shoot_torpedo_task.add_argument('timeout', type=float)
        run_shoot_torpedo_task.add_argument('frequency', type=float)
        run_shoot_torpedo_task.add_argument('distance', type=float)
        run_shoot_torpedo_task.add_argument('error_factor', type=float)
        run_shoot_torpedo_task.add_argument('error_threshold', type=float)
        run_shoot_torpedo_task.set_defaults(func=self._handle_run_shoot_torpedo_task)

        run_detect_pinger_task = subparsers.add_parser('run_detect_pinger_task')
        run_detect_pinger_task.add_argument('frequency', type=float)
        run_detect_pinger_task.add_argument('depth', type=float)
        run_detect_pinger_task.set_defaults(func=self._handle_run_detect_pinger_task)

        run_gate_task = subparsers.add_parser('run_gate_task')
        run_gate_task.set_defaults(func=self._handle_run_gate_task)

        cancel_task = subparsers.add_parser('cancel_task')
        cancel_task.set_defaults(func=self._handle_cancel_task)

        set_eater = subparsers.add_parser('set_eater')
        set_eater.add_argument('direction', type=float)
        set_eater.set_defaults(func=self._handle_set_eater)

        set_sphincter = subparsers.add_parser('set_sphincter')
        set_sphincter.add_argument('strength', type=float)
        set_sphincter.add_argument('duration', type=float)
        set_sphincter.add_argument('--open', default=False, action="store_true")
        set_sphincter.set_defaults(func=self._handle_set_sphincter)

        open_sphincter = subparsers.add_parser('open_sphincter')
        open_sphincter.set_defaults(func=self._handle_open_sphincter)

        close_sphincter = subparsers.add_parser('close_sphincter')
        close_sphincter.set_defaults(func=self._handle_close_sphincter)

        circle_buoy = subparsers.add_parser('circle_buoy')
        circle_buoy.add_argument('radius', type=float)
        circle_buoy.add_argument('latch', type=bool)
        circle_buoy.set_defaults(func=self.git)

        debug_depth = subparsers.add_parser('debug_depth')
        debug_depth.set_defaults(func=self._handle_debug_depth)

        approach_samples = subparsers.add_parser('approach_samples')
        approach_samples.set_defaults(func=self._handle_approach_samples)

        clear_map = subparsers.add_parser('clear_map')
        clear_map.set_defaults(func=self._handle_clear_map)

        return parser


def main():
    rospy.init_node('teleop_mission', log_level=rospy.ERROR)
    n = TeleopMission()
    n.start()
