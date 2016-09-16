# message(STATUS "Adding \"testing\"")

FILE(GLOB SOURCES_TEST *.test)

list(LENGTH SOURCES_TEST index)
message(STATUS "Found ${index} performance tests")


find_package(PythonInterp)
if (NOT ${PYTHONINTERP_FOUND})
  message("** Python is not found. Skipping performance tests")
  return()
endif()

set(CMAKE_INCLUDE_CURRENT_DIR ON)
cuda_include_directories(${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
cuda_include_directories(${CMAKE_SOURCE_DIR}/testing)
include_directories(${CMAKE_SOURCE_DIR}/testing)

set(compile_source "${CMAKE_CURRENT_BINARY_DIR}/compile_source.py")
FILE(WRITE ${compile_source}
  "import sys\n"
  "sys.path.append(\"${CMAKE_CURRENT_SOURCE_DIR}\")\n"
  "from build.perftest import compile_test\n"
  "compile_test(str(sys.argv[1]),str(sys.argv[2]))\n"
  )
set(targets "")
set(perf_sources "")
foreach(src ${SOURCES_TEST})
  get_filename_component(exec_name ${src} NAME_WE)
  set(target perf-${exec_name})
  set(dst ${CMAKE_CURRENT_BINARY_DIR}/${exec_name}.cu)
  add_custom_command(
    OUTPUT ${dst}
    DEPENDS ${src}
    COMMAND "${PYTHON_EXECUTABLE}" 
    ARGS ${compile_source}$ "" ${src} "" ${dst}$  "" ${dst}
    COMMENT "Generate perforfmance test \"${dst}\" from \"${src}\" "
    )
  set(cuda_src ${dst})
  thrust_add_executable(${target} ${cuda_src})
  set_target_properties(${target} PROPERTIES OUTPUT_NAME ${exec_name})
  install(TARGETS ${target} DESTINATION "performance/${HOST_BACKEND}_host_${DEVICE_BACKEND}_device_${THRUST_MODE}" OPTIONAL COMPONENT performance-bin)
  list(APPEND targets ${target})
  list(APPEND perf_sources ${cuda_src})
endforeach()

add_custom_target(performance-bin DEPENDS ${targets})
add_custom_target(install-performance-bin
  COMMAND 
      "${CMAKE_COMMAND}" -DCMAKE_INSTALL_COMPONENT=performance-bin
      -P "${CMAKE_BINARY_DIR}/cmake_install.cmake"
)

# install(FILES ${perf_sources} DESTINATION "performance" COMPONENT performance)

