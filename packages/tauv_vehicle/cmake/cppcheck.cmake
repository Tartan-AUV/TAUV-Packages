#additional target to perform cppcheck run, requires cppcheck
find_program(CPPCHECK cppcheck)
# get all c++ project files
file(GLOB_RECURSE ALL_SOURCE_FILES *.cpp *.h)

add_custom_target(
  cppcheck
  COMMAND ${CPPCHECK}
  --enable=all
  --std=c++11
  --language=c++
  --library=std.cfg
  -i /opt/ros/kinetic/include
  --template="[{severity}][{id}] {message} {callstack} \(On {file}:{line}\)"
  --verbose
  --quiet
  ${ALL_SOURCE_FILES}
)
