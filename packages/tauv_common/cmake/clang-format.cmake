#additional target to perform cppcheck run, requires cppcheck
find_program(CLANGFORMAT clang-format)
# get all c++ project files
file(GLOB_RECURSE ALL_SOURCE_FILES *.cpp *.h)

add_custom_target(
  clang-format
  COMMAND ${CLANGFORMAT}
  -style=Google
  -i
  ${ALL_SOURCE_FILES}
)

# For use in CI
add_custom_target(
  clang-format-check
  COMMAND clang-format -output-replacements-xml -style=Google ${ALL_SOURCE_FILES} | grep \"<replacement \"  && exit 1 || exit 0
)
