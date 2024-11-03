cmake_minimum_required(VERSION 2.8.3)
project(tauv_mission)

add_compile_options(-Wall -Werror -std=c++14)

find_package(catkin REQUIRED COMPONENTS
    std_msgs
    message_generation
    sensor_msgs
    geometry_msgs
    message_filters
    tf
    std_srvs
    tauv_msgs
  )

catkin_python_setup()

## Generate services in the 'srv' folder
#add_service_files(
#    FILES
#)

## Generate added messages and services with any dependencies listed here
#generate_messages(
#    DEPENDENCIES
#)

catkin_package(CATKIN_DEPENDS
    message_runtime
    std_msgs
    std_srvs
    roscpp
    rospy
    sensor_msgs
    geometry_msgs
    tf
    tauv_config
    message_filters
  )

include_directories(
        src
        include
        ${catkin_INCLUDE_DIRS}
)

#catkin_install_python(PROGRAMS scripts/thruster_allocator
#                      DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

install(DIRECTORY launch config include
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
        PATTERN "*~" EXCLUDE)
