# Check source code for issues that can be found by pattern matching:
add_test(
  NAME thrust.test.cmake.check_source_files
  COMMAND
    "${CMAKE_COMMAND}"
      -D "Thrust_SOURCE_DIR=${Thrust_SOURCE_DIR}"
      -P "${CMAKE_CURRENT_LIST_DIR}/check_source_files.cmake"
)
