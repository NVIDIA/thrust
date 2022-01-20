#! /usr/bin/env bash

# Copyright (c) 2018-2022 NVIDIA Corporation
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

# echo_and_run name args...
# Echo ${args[@]}, then execute ${args[@]}
function echo_and_run {
  echo "${1}: ${@:2}"
  ${@:2}
}

# echo_and_run_timed name args...
# Echo ${args[@]}, then execute ${args[@]} and report how long it took,
# including ${name} in the output of the time.
function echo_and_run_timed {
  echo "${@:2}"
  TIMEFORMAT=$'\n'"${1} Time: %lR"
  time ${@:2}
}

# join_delimit <delimiter> [value [value [...]]]
# Combine all values into a single string, separating each by a single character
# delimiter. Eg:
# foo=(bar baz kramble)
# joined_foo=$(join_delimit "|" "${foo[@]}")
# echo joined_foo # "bar|baz|kramble"
function join_delimit {
  local IFS="${1}"
  shift
  echo "${*}"
}

################################################################################
# VARIABLES - Set up bash and environmental variables.
################################################################################

# Get the variables the Docker container set up for us: ${CXX}, ${CUDACXX}, etc.
source /etc/cccl.bashrc

# Set path.
export PATH=/usr/local/cuda/bin:${PATH}

# Set home to the job's workspace.
export HOME=${WORKSPACE}

cd ${WORKSPACE}

# Remove any old .ninja_log file so the PrintNinjaBuildTimes step is accurate:
rm -f build/.ninja_log

if [[ -z "${CMAKE_BUILD_TYPE}" ]]; then
  CMAKE_BUILD_TYPE="Release"
fi

if [[ -z "${CMAKE_CONFIG_PRESET}" ]]; then
  echo "CMAKE_CONFIG_PRESET is undefined."
  cmake --list-presets
  exit 1
fi

if [[ -z "${CMAKE_TEST_PRESET}" ]]; then
  echo "CMAKE_TEST_PRESET is undefined."
  ctest --list-presets
  exit 1
fi

append CMAKE_FLAGS "--preset ${CMAKE_CONFIG_PRESET}"
append CMAKE_BUILD_FLAGS "--"
append CTEST_FLAGS "--preset ${CMAKE_TEST_PRESET}"

# The Docker image sets up `${CXX}` and `${CUDACXX}`.
append CMAKE_FLAGS "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
append CMAKE_FLAGS "-DCMAKE_CUDA_COMPILER='${CUDACXX}'"

if [[ "${CXX_TYPE}" == "nvcxx" ]]; then
  # NVC++ isn't properly detected by CMake, so we have to tell CMake to ignore
  # detection and explicit provide the compiler ID. Ninja currently isn't
  # supported, so we just use makefiles.
  append CMAKE_FLAGS "-DCMAKE_CUDA_COMPILER_FORCED=ON"
  append CMAKE_FLAGS "-DCMAKE_CUDA_COMPILER_ID=NVCXX"
  # We use NVC++ "slim" image which only contain a single CUDA toolkit version.
  # When using NVC++ in an environment without GPUs (like our CPU-only
  # builders) it unfortunately defaults to the oldest CUDA toolkit version it
  # supports, even if that version is not in the image. So, we have to
  # explicitly tell NVC++ it which CUDA toolkit version to use.
  CUDA_VER=$(echo ${SDK_VER} | sed 's/.*\(cuda[0-9]\+\.[0-9]\+\)/\1/')
  append CMAKE_FLAGS "-DCMAKE_CUDA_FLAGS=-gpu=${CUDA_VER}"
  # Don't stop on build failures.
  append CMAKE_BUILD_FLAGS "-k"
else
  if [[ "${CXX_TYPE}" == "icc" ]]; then
    # Only the latest version of the Intel C++ compiler, which NVCC doesn't
    # officially support yet, is freely available.
    append CMAKE_FLAGS "-DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler"
  fi
  # We're using NVCC so we need to set the host compiler.
  append CMAKE_FLAGS "-DCMAKE_CXX_COMPILER='${CXX}'"
  append CMAKE_FLAGS "-DCMAKE_CUDA_HOST_COMPILER='${CXX}'"
  append CMAKE_FLAGS "-G Ninja"
  # Don't stop on build failures.
  append CMAKE_BUILD_FLAGS "-k0"
fi

if [[ -n "${PARALLEL_LEVEL}" ]]; then
  DETERMINE_PARALLELISM_FLAGS="-j ${PARALLEL_LEVEL}"
fi

if [[ "${BUILD_TYPE}" == "cpu" ]] && [[ "${CXX_TYPE}" == "nvcxx" ]]; then
  # If no GPU is automatically detected, NVC++ insists that you explicitly
  # provide an architecture.
  # TODO: This logic should really be moved into CMake, but it will be
  # tricky to do that until CMake officially supports NVC++.
  append CMAKE_FLAGS "-DTHRUST_ENABLE_COMPUTE_80=ON"
fi

if [[ -n "${@}" ]]; then
  append CMAKE_BUILD_FLAGS "${@}"

  CTEST_INCLUSION_REGEX=$(join_delimit "|" "${@}")
  append CTEST_FLAGS "-R ^${CTEST_INCLUSION_REGEX[@]}$"
fi

# Export variables so they'll show up in the logs when we report the environment.
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

env | sort

log "Check versions..."

# We use sed and echo below to ensure there is always one and only trailing
# line following the output from each tool.

${CXX} --version 2>&1 | sed -Ez '$ s/\n*$/\n/'

echo

${CUDACXX} --version 2>&1 | sed -Ez '$ s/\n*$/\n/'

echo

cmake --version 2>&1 | sed -Ez '$ s/\n*$/\n/'

echo

if [[ "${BUILD_TYPE}" == "gpu" ]]; then
  nvidia-smi 2>&1 | sed -Ez '$ s/\n*$/\n/'
fi

################################################################################
# BUILD - Build Thrust and CUB examples and tests.
################################################################################

log "Configure Thrust and CUB..."

echo_and_run_timed "Configure" cmake --log-level=VERBOSE ${CMAKE_FLAGS}
configure_status=$?

log "Build Thrust and CUB..."

# ${PARALLEL_LEVEL} needs to be passed after we run
# determine_build_parallelism.bash, so it can't be part of ${CMAKE_BUILD_FLAGS}.
set +e # Don't stop on build failures.
echo_and_run_timed "Build" cmake --build build/ ${CMAKE_BUILD_FLAGS} -j ${PARALLEL_LEVEL}
build_status=$?
set -e

################################################################################
# TEST - Run Thrust and CUB examples and tests.
################################################################################

log "Test Thrust and CUB..."

set +e # Don't stop on test failures.
echo_and_run_timed "Test" ctest ${CTEST_FLAGS} -O build/ctest_log
test_status=$?
set -e

################################################################################
# COMPILE TIME INFO: Print the 20 longest running build steps (ninja only)
################################################################################

if [[ -f ".ninja_log" ]]; then
  log "Checking slowest build steps:"
  echo_and_run "CompileTimeInfo" cmake -P cmake/PrintNinjaBuildTimes.cmake | head -n 23
fi

################################################################################
# RUNTIME INFO: Print the 20 longest running test steps
################################################################################

if [[ -f "ctest_log" ]]; then
  log "Checking slowest test steps:"
  echo_and_run "TestTimeInfo" cmake -DLOGFILE=build/ctest_log -P cmake/PrintCTestRunTimes.cmake | head -n 20
fi

################################################################################
# SUMMARY - Print status of each step and exit with failure if needed.
################################################################################

log "Summary:"
echo "- Configure Error Code: ${configure_status}"
echo "- Build Error Code: ${build_status}"
echo "- Test Error Code: ${test_status}"

if [[ "${configure_status}" != "0" ]] || \
   [[ "${build_status}" != "0" ]] || \
   [[ "${test_status}" != "0" ]]; then
     cd build # Enter build directory for debugging convenience
     exit 1
fi
