import rospy
from typing import Optional
from threading import Lock, Event, Thread
from tasks.task import Task, TaskResources, TaskStatus, TaskResult
import missions
from missions.mission import Mission
from missions.manifest import get_mission_by_name
from std_srvs.srv import Trigger, TriggerRequest, TriggerResponse
from tauv_msgs.srv import RunMission, RunMissionRequest, RunMissionResponse
from motion_client.motion_client import MotionClient
from actuator_client import ActuatorClient
from transform_client import TransformClient
from map_client import MapClient


class MissionManager:

    def __init__(self):
        self._lock: Lock = Lock()

        self._mission: Optional[Mission] = None

        self._task: Optional[Task] = None
        self._task_result: Optional[TaskResult] = None

        self._task_resources: TaskResources = TaskResources(
            motion=MotionClient(),
            actuators=ActuatorClient(),
            transforms=TransformClient(),
            map=MapClient(),
        )

        self._mission_start_event: Event = Event()
        self._mission_cancel_event: Event = Event()
        self._mission_update_event: Event = Event()

        self._task_done_event: Event = Event()
        self._task_update_event: Event = Event()

        self._run_mission_service: rospy.Service = rospy.Service("mission/run", RunMission, self._handle_run_mission)
        self._cancel_mission_service: rospy.Service = rospy.Service("mission/cancel", Trigger, self._handle_cancel_mission)

        self._spin_mission_thread: Optional[Thread] = None
        self._spin_task_thread: Optional[Thread] = None
        self._task_thread: Optional[Thread] = None

    def run(self) -> None:
        self._spin_mission_thread = Thread(target=self._spin_mission, daemon=True)
        self._spin_mission_thread.start()
        self._spin_task_thread = Thread(target=self._spin_task, daemon=True)
        self._spin_task_thread.start()
        rospy.spin()

    def _spin_mission(self):
        while True:
            rospy.logdebug("_spin_mission waiting")
            self._mission_update_event.wait()

            if self._mission_start_event.is_set():
                self._start_mission()
            elif self._mission_cancel_event.is_set():
                self._cancel_mission()

            self._mission_update_event.clear()

    def _spin_task(self):
        while True:
            rospy.logdebug("_spin_task waiting")
            self._task_update_event.wait()

            self._task_update_event.clear()

            if self._task_done_event.is_set():
                self._task_done_event.clear()

                self._transition_task()

    def _start_mission(self):
        rospy.logdebug("_start_mission acquiring lock")
        with self._lock:
            rospy.logdebug("_start_mission acquired lock")
            self._task = self._mission.entrypoint()
            self._task_thread = Thread(target=self._run_task, daemon=True)
            self._task_thread.start()

            self._mission_start_event.clear()

    def _cancel_mission(self):
        rospy.logdebug('_cancel_mission acquiring lock')
        with self._lock:
            rospy.logdebug('_cancel_mission acquired lock')
            self._mission = None

            if self._task is not None:
                self._task.cancel()

    def _transition_task(self):
        rospy.logdebug('_transition_task acquiring lock')
        with self._lock:
            rospy.logdebug('_transition_task acquired lock')
            if self._mission_cancel_event.is_set():
                rospy.logdebug('_transition_task _mission_cancel_event set, short-circuiting')
                self._mission_cancel_event.clear()
                return

            new_task = self._mission.transition(self._task, self._task_result)

            if new_task is None:
                rospy.logdebug('_transition_task new_task is None, mission complete')
                self._mission = None
                self._task = None
                self._task_result = None
                return

            rospy.logdebug(f'_transition_task running new_task {type(new_task)}')
            self._task = new_task
            self._task_result = None
            self._task_thread = Thread(target=self._run_task, daemon=True)
            self._task_thread.start()

    def _run_task(self):
        rospy.logdebug(f'_run_task running self._task {type(self._task)}')
        task_result = self._task.run(self._task_resources)
        rospy.logdebug(f'_run_task done running self._task {type(self._task)}')
        rospy.logdebug('_run_task acquiring lock')
        with self._lock:
            rospy.logdebug('_run_task acquired lock')
            self._task_result = task_result
        self._task_done_event.set()
        self._task_update_event.set()

    def _handle_run_mission(self, req: RunMissionRequest) -> RunMissionResponse:
        rospy.logdebug('_handle_run_mission acquiring lock')
        with self._lock:
            rospy.logdebug('_handle_run_mission acquired lock')
            res = RunMissionResponse()

            if self._mission is not None:
                res.success = False
                res.message = 'mission in progress'
                return res

            mission = get_mission_by_name(req.mission_name)

            if mission is None:
                rospy.logdebug('_handle_run_mission mission is None, short-circuiting')
                res.success = False
                res.message = f'could not find mission named {req.mission_name}'
                return res

            self._mission = mission()

            rospy.logdebug('_handle_run_mission setting mission_start_event')
            self._mission_start_event.set()
            self._mission_update_event.set()

            res.success = True
            return res

    def _handle_cancel_mission(self, req: TriggerRequest) -> TriggerResponse:
        rospy.logdebug('_handle_cancel_mission acquiring lock')
        with self._lock:
            rospy.logdebug('_handle_cancel_mission acquired lock')
            res = TriggerResponse()

            if self._mission is None:
                rospy.logdebug('_handle_cancel_mission self._mission is None, short-circuiting')
                res.success = False
                res.message = 'no mission in progress'
                return res

            rospy.logdebug('_handle_cancel_mission setting mission cancel event')
            self._mission_cancel_event.set()
            self._mission_update_event.set()

            res.success = True
            return res


def main():
    rospy.init_node('mission_manager', log_level=rospy.DEBUG)

    node = MissionManager()
    node.run()