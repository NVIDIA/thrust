#! /usr/bin/env bash

# Copyright (c) 2018-2020 NVIDIA Corporation
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Released under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.

################################################################################
# Thrust and CUB build script for gpuCI
################################################################################

set -e

# append variable value
# Appends ${value} to ${variable}, adding a space before ${value} if
# ${variable} is not empty.
function append {
  tmp="${!1:+${!1} }${2}"
  eval "${1}=\${tmp}"
}

# log args...
# Prints out ${args[*]} with a gpuCI log prefix and a newline before and after.
function log() {
  printf "\n>>>> %s\n\n" "${*}"
}

# print_with_trailing_blank_line args...
# Prints ${args[*]} with one blank line following, preserving newlines within
# ${args[*]} but stripping any preceding ${args[*]}.
function print_with_trailing_blank_line {
  printf "%s\n\n" "${*}"
}

# echo_and_run_timed name args...
# Echo ${args[@]}, then execute ${args[@]} and report how long it took,
# including ${name} in the output of the time.
function echo_and_run_timed {
  echo "${@:2}"
  TIMEFORMAT=$'\n'"${1} Time: %lR"
  time ${@:2}
}

################################################################################
# VARIABLES - Set up bash and environmental variables.
################################################################################

# Get the variables the Docker container set up for us: ${CXX}, ${CUDACXX}, etc.
source /etc/cccl.bashrc

# Set path and build parallel level
export PATH=/usr/local/cuda/bin:${PATH}

# Set home to the job's workspace.
export HOME=${WORKSPACE}

# Switch to the build directory.
cd ${WORKSPACE}
mkdir -p build
cd build

if [[ -z "${CMAKE_BUILD_TYPE}" ]]; then
  CMAKE_BUILD_TYPE="Release"
fi

CMAKE_BUILD_FLAGS="--"

# The Docker image sets up `${CXX}` and `${CUDACXX}`.
append CMAKE_FLAGS "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
append CMAKE_FLAGS "-DCMAKE_CUDA_COMPILER='${CUDACXX}'"

if [[ "${CXX_TYPE}" == "nvcxx" ]]; then
  # NVC++ isn't properly detected by CMake, so we have to tell CMake to ignore
  # detection and explicit provide the compiler ID. Ninja currently isn't
  # supported, so we just use makefiles.
  append CMAKE_FLAGS "-DCMAKE_CUDA_COMPILER_FORCED=ON"
  append CMAKE_FLAGS "-DCMAKE_CUDA_COMPILER_ID=NVCXX"
  # Don't stop on build failures.
  append CMAKE_BUILD_FLAGS "-k"
  # NVC++ currently uses a lot of memory.
  PARALLEL_LEVEL=1
else
  if [[ "${CXX_TYPE}" == "icc" ]]; then
    # Only the latest version of the Intel C++ compiler, which NVCC doesn't
    # officially support yet, is freely available.
    append CMAKE_FLAGS "-DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler"
  fi
  # We're using NVCC so we need to set the host compiler.
  append CMAKE_FLAGS "-DCMAKE_CXX_COMPILER='${CXX}'"
  append CMAKE_FLAGS "-G Ninja"
  # Don't stop on build failures.
  append CMAKE_BUILD_FLAGS "-k0"
fi

if [[ -n "${PARALLEL_LEVEL}" ]]; then
  DETERMINE_PARALLELISM_FLAGS="-j ${PARALLEL_LEVEL}"
fi

# COVERAGE_PLAN options:
# * Exhaustive
# * Thorough
# * Minimal
if [[ -z "${COVERAGE_PLAN}" ]]; then
  # `ci/local/build.bash` always sets a coverage plan, so we can assume we're
  # in gpuCI if one was not set.
  if [[ "${CXX_TYPE}" == "nvcxx" ]]; then
    # Today, NVC++ builds take too long to do anything more than Minimal.
    COVERAGE_PLAN="Minimal"
  elif [[ "${BUILD_TYPE}" == "cpu" ]] && [[ "${BUILD_MODE}" == "branch" ]]; then
    # Post-commit CPU CI builds.
    COVERAGE_PLAN="Exhaustive"
  elif [[ "${BUILD_TYPE}" == "cpu" ]]; then
    # Pre-commit CPU CI builds.
    COVERAGE_PLAN="Thorough"
  elif [[ "${BUILD_TYPE}" == "gpu" ]]; then
    # Pre- and post-commit GPU CI builds.
    COVERAGE_PLAN="Minimal"
  fi
fi

case "${COVERAGE_PLAN}" in
  Exhaustive)
    append CMAKE_FLAGS "-DTHRUST_ENABLE_MULTICONFIG=ON"
    append CMAKE_FLAGS "-DTHRUST_MULTICONFIG_ENABLE_DIALECT_CPP11=ON"
    append CMAKE_FLAGS "-DTHRUST_IGNORE_DEPRECATED_CPP_11=ON"
    append CMAKE_FLAGS "-DTHRUST_MULTICONFIG_ENABLE_DIALECT_CPP14=ON"
    append CMAKE_FLAGS "-DTHRUST_MULTICONFIG_ENABLE_DIALECT_CPP17=OFF"
    append CMAKE_FLAGS "-DTHRUST_MULTICONFIG_ENABLE_SYSTEM_CPP=ON"
    append CMAKE_FLAGS "-DTHRUST_MULTICONFIG_ENABLE_SYSTEM_TBB=ON"
    append CMAKE_FLAGS "-DTHRUST_MULTICONFIG_ENABLE_SYSTEM_OMP=ON"
    append CMAKE_FLAGS "-DTHRUST_MULTICONFIG_ENABLE_SYSTEM_CUDA=ON"
    append CMAKE_FLAGS "-DTHRUST_INCLUDE_CUB_CMAKE=ON"
    append CMAKE_FLAGS "-DTHRUST_MULTICONFIG_WORKLOAD=LARGE"
    ;;
  Thorough)
    append CMAKE_FLAGS "-DTHRUST_ENABLE_MULTICONFIG=ON"
    append CMAKE_FLAGS "-DTHRUST_MULTICONFIG_ENABLE_DIALECT_CPP11=ON"
    append CMAKE_FLAGS "-DTHRUST_IGNORE_DEPRECATED_CPP_11=ON"
    append CMAKE_FLAGS "-DTHRUST_MULTICONFIG_ENABLE_DIALECT_CPP14=ON"
    append CMAKE_FLAGS "-DTHRUST_MULTICONFIG_ENABLE_DIALECT_CPP17=OFF"
    append CMAKE_FLAGS "-DTHRUST_MULTICONFIG_ENABLE_SYSTEM_CPP=ON"
    append CMAKE_FLAGS "-DTHRUST_MULTICONFIG_ENABLE_SYSTEM_TBB=ON"
    append CMAKE_FLAGS "-DTHRUST_MULTICONFIG_ENABLE_SYSTEM_OMP=ON"
    append CMAKE_FLAGS "-DTHRUST_MULTICONFIG_ENABLE_SYSTEM_CUDA=ON"
    append CMAKE_FLAGS "-DTHRUST_MULTICONFIG_WORKLOAD=SMALL"
    append CMAKE_FLAGS "-DTHRUST_INCLUDE_CUB_CMAKE=ON"
    append CMAKE_FLAGS "-DTHRUST_AUTO_DETECT_COMPUTE_ARCHS=ON"
    if [[ "${CXX_TYPE}" != "nvcxx" ]]; then
      # NVC++ can currently only target one compute architecture at a time.
      append CMAKE_FLAGS "-DTHRUST_ENABLE_COMPUTE_50=ON"
      append CMAKE_FLAGS "-DTHRUST_ENABLE_COMPUTE_60=ON"
      append CMAKE_FLAGS "-DTHRUST_ENABLE_COMPUTE_70=ON"
    fi
    append CMAKE_FLAGS "-DTHRUST_ENABLE_COMPUTE_80=ON"
    ;;
  Minimal)
    append CMAKE_FLAGS "-DTHRUST_ENABLE_MULTICONFIG=ON"
    append CMAKE_FLAGS "-DTHRUST_MULTICONFIG_ENABLE_DIALECT_CPP11=OFF"
    append CMAKE_FLAGS "-DTHRUST_MULTICONFIG_ENABLE_DIALECT_CPP14=ON"
    append CMAKE_FLAGS "-DTHRUST_MULTICONFIG_ENABLE_DIALECT_CPP17=OFF"
    append CMAKE_FLAGS "-DTHRUST_MULTICONFIG_ENABLE_SYSTEM_CPP=ON"
    append CMAKE_FLAGS "-DTHRUST_MULTICONFIG_ENABLE_SYSTEM_TBB=OFF"
    append CMAKE_FLAGS "-DTHRUST_MULTICONFIG_ENABLE_SYSTEM_OMP=OFF"
    append CMAKE_FLAGS "-DTHRUST_MULTICONFIG_ENABLE_SYSTEM_CUDA=ON"
    append CMAKE_FLAGS "-DTHRUST_MULTICONFIG_WORKLOAD=SMALL"
    append CMAKE_FLAGS "-DTHRUST_INCLUDE_CUB_CMAKE=ON"
    append CMAKE_FLAGS "-DCUB_ENABLE_THOROUGH_TESTING=OFF"
    append CMAKE_FLAGS "-DCUB_ENABLE_BENCHMARK_TESTING=OFF"
    append CMAKE_FLAGS "-DCUB_ENABLE_MINIMAL_TESTING=ON"
    append CMAKE_FLAGS "-DTHRUST_AUTO_DETECT_COMPUTE_ARCHS=ON"
    if [[ "${BUILD_TYPE}" == "cpu" ]] && [[ "${CXX_TYPE}" == "nvcxx" ]]; then
      # If no GPU is automatically detected, NVC++ insists that you explicitly
      # provide an architecture.
      # TODO: This logic should really be moved into CMake, but it will be
      # tricky to do that until CMake officially supports NVC++.
      append CMAKE_FLAGS "-DTHRUST_ENABLE_COMPUTE_80=ON"
    fi
    ;;
esac

if [[ -n "${@}" ]]; then
  append CMAKE_BUILD_FLAGS "${@}"
fi

append CTEST_FLAGS "--output-on-failure"

if [[ "${BUILD_TYPE}" == "cpu" ]]; then
  append CTEST_FLAGS "-E ^cub|^thrust.*cuda"
fi

if [[ -n "${@}" ]]; then
  for arg in "${@}"
  do
    append CTEST_FLAGS "-R ^${arg}$"
  done
fi

# Export variables so they'll show up in the logs when we report the environment.
export COVERAGE_PLAN
export CMAKE_FLAGS
export CMAKE_BUILD_FLAGS
export CTEST_FLAGS

################################################################################
# ENVIRONMENT - Configure and print out information about the environment.
################################################################################

log "Determine system topology..."

# Set `${PARALLEL_LEVEL}` if it is unset; otherwise, this just reports the
# system topology.
source ${WORKSPACE}/ci/common/determine_build_parallelism.bash ${DETERMINE_PARALLELISM_FLAGS}

log "Get environment..."

env

log "Check versions..."

# We use sed and echo below to ensure there is always one and only trailing
# line following the output from each tool.

${CXX} --version 2>&1 | sed -Ez '$ s/\n*$/\n/'

echo

${CUDACXX} --version 2>&1 | sed -Ez '$ s/\n*$/\n/'

echo

if [[ "${BUILD_TYPE}" == "gpu" ]]; then
  nvidia-smi 2>&1 | sed -Ez '$ s/\n*$/\n/'
fi

################################################################################
# BUILD - Build Thrust and CUB examples and tests.
################################################################################

log "Configure Thrust and CUB..."

echo_and_run_timed "Configure" cmake .. ${CMAKE_FLAGS}

log "Build Thrust and CUB..."

# ${PARALLEL_LEVEL} needs to be passed after we run
# determine_build_parallelism.bash, so it can't be part of ${CMAKE_BUILD_FLAGS}.
set +e # Don't stop on build failures.
echo_and_run_timed "Build" cmake --build . ${CMAKE_BUILD_FLAGS} -j ${PARALLEL_LEVEL}
set -e

################################################################################
# TEST - Run Thrust and CUB examples and tests.
################################################################################

log "Test Thrust and CUB..."

echo_and_run_timed "Test" ctest ${CTEST_FLAGS}

