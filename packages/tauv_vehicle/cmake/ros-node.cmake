# Macro for declaring ros nodes
# First argument is the name, second is the file with the main function
# Any other arguments are considered library dependencies
macro(node name file)
    add_executable(${name} ${file})
    add_dependencies(${name} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
    target_link_libraries(${name} ${ARGN} ${catkin_LIBRARIES})
endmacro(node)

# Macro for declaring a library
# First argument is the name of the library. All others are the source files for
# the library
macro(library name ${file})
  add_library(${name} ${file} ${ARGN})
  add_dependencies(${name} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
endmacro(library)
