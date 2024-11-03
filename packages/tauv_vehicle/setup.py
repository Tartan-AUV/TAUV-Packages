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

# fetch values from package.xml
setup_args = generate_distutils_setup(
    packages=['depth_sensor', 'thrusters', 'battery', 'sonar', 'xsens_imu_sync', 'teledyne_dvl', 'arduino', 'oakd', 'pinger_localizer'],
    package_dir={'': 'src'},
    requires=['rospy'],
    scripts=['scripts/depth_sensor',
             'scripts/thrusters',
             'scripts/battery',
             'scripts/sonar',
             'scripts/xsens_imu_sync',
             'scripts/teledyne_dvl',
             'scripts/arduino',
             'scripts/oakd',
             'scripts/pinger_localizer',
             ]
)

setup(**setup_args)
