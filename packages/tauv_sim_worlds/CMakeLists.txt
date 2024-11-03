cmake_minimum_required(VERSION 2.8.3)
project(tauv_sim_worlds)

find_package(catkin REQUIRED COMPONENTS gazebo_ros)

catkin_package()

install(DIRECTORY launch worlds models
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
    PATTERN "*~" EXCLUDE)
