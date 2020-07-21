# Support adding Thrust to a parent project via add_subdirectory.
# See examples/cmake/add_subdir/CMakeLists.txt for details.
if (NOT "${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_CURRENT_LIST_DIR}")
  include(cmake/ThrustAddSubdir.cmake)
  return()
endif()

# 3.15 is the minimum.
# 3.17 for nvc++/Feta
# 3.18 for C++17 + CUDA
cmake_minimum_required(VERSION 3.15)

# Remove this when we use the new CUDA_ARCHITECTURES properties with both
# nvcc and nvc++.
if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.18)
  cmake_policy(SET CMP0104 OLD)
endif()

project(Thrust NONE)

include(cmake/AppendOptionIfAvailable.cmake)

include(cmake/ThrustBuildCompilerTargets.cmake)
include(cmake/ThrustBuildTargetList.cmake)
include(cmake/ThrustMultiConfig.cmake)
include(cmake/ThrustInstallRules.cmake)
include(cmake/ThrustUtilities.cmake)

option(THRUST_ENABLE_HEADER_TESTING "Test that all public headers compile." "ON")
option(THRUST_ENABLE_TESTING "Build Thrust testing suite." "ON")
option(THRUST_ENABLE_EXAMPLES "Build Thrust examples." "ON")
option(THRUST_INCLUDE_CUB_CMAKE "Build CUB tests and examples. (Requires CUDA)." "OFF")

# Check if we're actually building anything before continuing. If not, no need
# to search for deps, etc. This is a common approach for packagers that just
# need the install rules. See GH issue thrust/thrust#1211.
if (NOT (THRUST_ENABLE_HEADER_TESTING OR
         THRUST_ENABLE_TESTING OR
         THRUST_ENABLE_EXAMPLES OR
         THRUST_INCLUDE_CUB_CMAKE))
  return()
endif()

# Add cache string options for CMAKE_BUILD_TYPE and default to RelWithDebInfo.
if ("" STREQUAL "${CMAKE_BUILD_TYPE}")
  set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Choose the type of build." FORCE)

  set_property(
    CACHE CMAKE_BUILD_TYPE
    PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel
  )
endif ()

# Disable compiler extensions:
set(CMAKE_CXX_EXTENSIONS OFF)

# Where to put build outputs. Use CMAKE_BINARY_DIR so they'll show up in the
# top-level project's dir when building Thrust via add_subdirectory.
set(THRUST_LIBRARY_OUTPUT_DIR "${CMAKE_BINARY_DIR}/lib")
set(THRUST_EXECUTABLE_OUTPUT_DIR "${CMAKE_BINARY_DIR}/bin")

# Temporary hacks to make Feta work; this requires you to define
# `CMAKE_CUDA_COMPILER_ID=Feta` and `CMAKE_CUDA_COMPILER_FORCED`.
if ("Feta" STREQUAL "${CMAKE_CUDA_COMPILER_ID}")
  # If using Feta, don't set CXX compiler
  if (NOT "${CMAKE_CXX_COMPILER}" STREQUAL "")
    unset(CMAKE_CXX_COMPILER CACHE)
    message(FATAL_ERROR "You are using Feta as your CUDA C++ compiler, but have"
      " specified a different ISO C++ compiler; Feta acts as both, so please"
      " unset the CMAKE_CXX_COMPILER variable.")
  endif ()

  # We don't set CMAKE_CUDA_HOST_COMPILER for Feta; if we do, CMake tries to
  # pass `-ccbin ${CMAKE_CUDA_HOST_COMPILER}` to Feta, which it doesn't
  # understand.
  if (NOT "${CMAKE_CUDA_HOST_COMPILER}" STREQUAL "")
    unset(CMAKE_CUDA_HOST_COMPILER CACHE)
    message(FATAL_ERROR "You are using Feta as your CUDA C++ compiler, but have"
      " specified a different host ISO C++ compiler; Feta acts as both, so"
      " please unset the CMAKE_CUDA_HOST_COMPILER variable.")
  endif ()

  set(CMAKE_CXX_COMPILER "${CMAKE_CUDA_COMPILER}")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -stdpar")
  set(CMAKE_CUDA_HOST_LINK_LAUNCHER "${CMAKE_CUDA_COMPILER}")
  set(CMAKE_CUDA_LINK_EXECUTABLE
      "<CMAKE_CUDA_HOST_LINK_LAUNCHER> ${CMAKE_CUDA_FLAGS} <LINK_FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES>")
endif ()

# This must appear after any changes to CMAKE_CXX_COMPILER or else CMake will
# delete the cache and reconfigure from scratch.
enable_language(CXX)

# We don't set CMAKE_CUDA_HOST_COMPILER for Feta; if we do, CMake tries to
# pass `-ccbin ${CMAKE_CUDA_HOST_COMPILER}` to Feta, which it doesn't
# understand.
if (NOT "Feta" STREQUAL "${CMAKE_CUDA_COMPILER_ID}")
  if (NOT ("${CMAKE_CUDA_HOST_COMPILER}" STREQUAL "" OR
           "${CMAKE_CUDA_HOST_COMPILER}" STREQUAL "${CMAKE_CXX_COMPILER}"))
    unset(CMAKE_CUDA_HOST_COMPILER CACHE)
    message(FATAL_ERROR "Thrust tests and examples require the C++ compiler"
      " and the CUDA host compiler to be the same; to set this compiler, please"
      " use the CMAKE_CXX_COMPILER variable, not the CMAKE_CUDA_HOST_COMPILER"
      " variable.")
  endif ()
  set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}")
endif ()

# Temporary hacks to make Feta work; this requires you to define
# `CMAKE_CUDA_COMPILER_ID=Feta` and `CMAKE_CUDA_COMPILER_FORCED`.
if ("Feta" STREQUAL "${CMAKE_CUDA_COMPILER_ID}")
  # Need 3.17 for the properties used below.
  cmake_minimum_required(VERSION 3.17)

  set(CMAKE_CUDA_STANDARD_DEFAULT 03)

  set(CMAKE_CUDA03_STANDARD_COMPILE_OPTION "-std=c++03")
  set(CMAKE_CUDA03_EXTENSION_COMPILE_OPTION "-std=c++03")
  set(CMAKE_CUDA03_STANDARD__HAS_FULL_SUPPORT TRUE)
  set_property(GLOBAL PROPERTY CMAKE_CUDA03_KNOWN_FEATURES)

  set(CMAKE_CUDA11_STANDARD_COMPILE_OPTION "-std=c++11")
  set(CMAKE_CUDA11_EXTENSION_COMPILE_OPTION "-std=c++11")
  set(CMAKE_CUDA11_STANDARD__HAS_FULL_SUPPORT TRUE)
  set_property(GLOBAL PROPERTY CMAKE_CUDA11_KNOWN_FEATURES)

  set(CMAKE_CUDA14_STANDARD_COMPILE_OPTION "-std=c++14")
  set(CMAKE_CUDA14_EXTENSION_COMPILE_OPTION "-std=c++14")
  set(CMAKE_CUDA14_STANDARD__HAS_FULL_SUPPORT TRUE)
  set_property(GLOBAL PROPERTY CMAKE_CUDA14_KNOWN_FEATURES)

  set(CMAKE_CUDA17_STANDARD_COMPILE_OPTION "-std=c++17")
  set(CMAKE_CUDA17_EXTENSION_COMPILE_OPTION "-std=c++17")
  set(CMAKE_CUDA17_STANDARD__HAS_FULL_SUPPORT TRUE)
  set_property(GLOBAL PROPERTY CMAKE_CUDA17_KNOWN_FEATURES)

  cmake_record_cuda_compile_features()

  set(CMAKE_CUDA_COMPILE_FEATURES
    ${CMAKE_CUDA03_COMPILE_FEATURES}
    ${CMAKE_CUDA11_COMPILE_FEATURES}
    ${CMAKE_CUDA14_COMPILE_FEATURES}
    ${CMAKE_CUDA17_COMPILE_FEATURES}
    ${CMAKE_CUDA20_COMPILE_FEATURES}
  )
endif ()

thrust_configure_multiconfig()
thrust_build_target_list()

thrust_update_system_found_flags()
message(STATUS "CPP system found?  ${THRUST_CPP_FOUND}")
message(STATUS "CUDA system found? ${THRUST_CUDA_FOUND}")
message(STATUS "TBB system found?  ${THRUST_TBB_FOUND}")
message(STATUS "OMP system found?  ${THRUST_OMP_FOUND}")

if (THRUST_CUDA_FOUND)
  include(cmake/ThrustCudaConfig.cmake)
endif()

if (THRUST_ENABLE_HEADER_TESTING)
  include(cmake/ThrustHeaderTesting.cmake)
endif()

# Both testing and examples use ctest
if (THRUST_ENABLE_TESTING OR THRUST_ENABLE_EXAMPLES)
  include(CTest)
  enable_testing()
endif()

if (THRUST_ENABLE_TESTING)
  add_subdirectory(testing)
endif()

if (THRUST_ENABLE_EXAMPLES)
  add_subdirectory(examples)
endif()

if (THRUST_INCLUDE_CUB_CMAKE AND THRUST_CUDA_FOUND)
  set(CUB_IN_THRUST ON)
  add_subdirectory(dependencies/cub)
endif()
