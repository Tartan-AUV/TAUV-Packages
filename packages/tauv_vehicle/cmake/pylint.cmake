#additional target to perform cppcheck run, requires cppcheck
find_program(PYLINT pylint PATHS /usr/bin/pylint)
# get all python project files
file(GLOB_RECURSE ALL_SOURCE_FILES *.py)

add_custom_target(
  pylint
  COMMAND env PYLINTHOME=${CMAKE_CURRENT_SOURCE_DIR} 
  ${PYLINT}
  --rcfile ${CMAKE_CURRENT_SOURCE_DIR}/.pylintrc
  ${ALL_SOURCE_FILES}
)
