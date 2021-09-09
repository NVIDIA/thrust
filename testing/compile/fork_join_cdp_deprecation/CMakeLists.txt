cmake_minimum_required(VERSION 3.15)

# Remove this when we use the new CUDA_ARCHITECTURES properties with both
# nvcc and nvc++.
if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.18)
  cmake_policy(SET CMP0104 OLD)
endif()

project(fork_join_cdp_deprecation CXX CUDA)

message(STATUS "Thrust_DIR=${Thrust_DIR}")

# Need C++14 for deprecation messages
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

find_package(Thrust)
thrust_create_target(thrust)

if (EXEC_SPACE STREQUAL "HOST")
  add_executable(main main_host.cu)
elseif(EXEC_SPACE STREQUAL "DEVICE")
  add_executable(main main_device.cu)
else()
  message(FATAL_ERROR "Invalid EXEC_SPACE: ${EXEC_SPACE}")
endif()

target_link_libraries(main thrust)

if (DIRECT_CALL)
  target_compile_definitions(main PUBLIC "DIRECT_CALL")
endif()

target_compile_definitions(main PUBLIC "EXEC_POLICY=${EXEC_POLICY}")

if (IGNORE_DEPRECATION)
  target_compile_definitions(main PUBLIC THRUST_IGNORE_DEPRECATED_FORK_JOIN_CDP)
endif()

if (ENABLE_RDC)
  if ("NVCXX" STREQUAL "${CMAKE_CUDA_COMPILER_ID}")
    set_target_properties(main PROPERTIES
      COMPILE_FLAGS "-gpu=rdc"
    )
  else()
    set_target_properties(main PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON
    )
  endif()
endif()

enable_testing()
add_test(NAME main COMMAND $<TARGET_FILE:main>)
