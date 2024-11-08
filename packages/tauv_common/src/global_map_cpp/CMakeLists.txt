set (CMAKE_CXX_STANDARD 17)

add_subdirectory(hungarian-algorithm-cpp)
add_subdirectory(kalman-filter)

library(global_map feature.cpp global_map.cpp)
node(global_map_cpp main.cpp global_map hungarian-matching kalman-filter)