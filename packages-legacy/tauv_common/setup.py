# Copyright (c) 2016 The UUV Simulator Authors.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

## ! DO NOT MANUALLY INVOKE THIS setup.py, USE CATKIN INSTEAD

from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup
from setuptools import find_packages

# fetch values from package.xml
setup_args = generate_distutils_setup(
    packages=['thruster_managers.models',
              'controllers',
              'controllers.controller',
              'planners',
              'planners.keyboard_planner',
              'planners.teleop_planner',
              'planners.mpc_planner',
              'planners.pid_planner',
              'state_estimation',
              'depth_estimation',
              'sonar',
              'vision',
              'vision.detectors',
              'teleop',
              'transform_manager',
              'thruster_manager',
              'grad_dynamics_parameter_estimator',
              'spkf_dynamics_parameter_estimator',
              'dynamics',
              'motion',
              'transform_client',
              'tauv_alarms',
              'tauv_util',
              'motion_client',
              'tauv_messages',
              'trajectories',
              'albatross_state_estimation'],
    # packages=find_packages(),
    package_dir={'': 'src'},
    requires=['rospy'],
    scripts=['scripts/thruster_allocator',
             'scripts/keyboard_planner',
             'scripts/teleop_planner',
             'scripts/controller',
             'scripts/mpc_planner',
             'scripts/depth_estimation',
             'scripts/state_estimation',
             'scripts/thruster_manager',
             'scripts/grad_dynamics_parameter_estimator',
             'scripts/spkf_dynamics_parameter_estimator',
             'scripts/alarm_server',
             'scripts/detector_bucket',
             'scripts/log_detections',
             'scripts/message_printer',
             'scripts/retare_sub_position',
             'scripts/watchdogs',
             'scripts/darknet_transformer',
             'scripts/bucket_to_tf',
             'scripts/shape_detector',
             'scripts/albatross_state_estimation'],

)
setup(**setup_args)
