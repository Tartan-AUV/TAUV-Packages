cmake_minimum_required(VERSION 2.8.3)
project(tauv_sim)

add_compile_options(-Wall -Werror -std=c++14)

find_package(catkin)

catkin_python_setup()

catkin_package()

include_directories(
  src
  ${catkin_INCLUDE_DIRS}
)

install(DIRECTORY launch
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
        PATTERN "*~" EXCLUDE)
