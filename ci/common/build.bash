#! /usr/bin/env bash

# Copyright (c) 2018-2020 NVIDIA Corporation
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Released under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.

################################################################################
# Thrust and CUB build script for gpuCI
################################################################################

set -e

# Logger function for build status output
function logger() {
  echo -e "\n>>>> ${@}\n"
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

# The Docker image sets up `${CXX}` and `${CUDACXX}`.
CMAKE_FLAGS="-G Ninja -DCMAKE_CXX_COMPILER='${CXX}' -DCMAKE_CUDA_COMPILER='${CUDACXX}'"

if [ "${BUILD_MODE}" == "branch" ]; then
  # Post-commit build.
  CMAKE_FLAGS="${CMAKE_FLAGS} -DTHRUST_INCLUDE_CUB_CMAKE=ON"
  CMAKE_FLAGS="${CMAKE_FLAGS} -DTHRUST_ENABLE_MULTICONFIG=ON"
  CMAKE_FLAGS="${CMAKE_FLAGS} -DTHRUST_MULTICONFIG_ENABLE_DIALECT_CPP11=ON"
  CMAKE_FLAGS="${CMAKE_FLAGS} -DTHRUST_IGNORE_DEPRECATED_CPP_11=ON"
  CMAKE_FLAGS="${CMAKE_FLAGS} -DTHRUST_MULTICONFIG_ENABLE_DIALECT_CPP14=ON"
  CMAKE_FLAGS="${CMAKE_FLAGS} -DTHRUST_MULTICONFIG_ENABLE_DIALECT_CPP17=OFF"
  CMAKE_FLAGS="${CMAKE_FLAGS} -DTHRUST_MULTICONFIG_ENABLE_SYSTEM_CPP=ON"
  CMAKE_FLAGS="${CMAKE_FLAGS} -DTHRUST_MULTICONFIG_ENABLE_SYSTEM_TBB=ON"
  CMAKE_FLAGS="${CMAKE_FLAGS} -DTHRUST_MULTICONFIG_ENABLE_SYSTEM_OMP=ON"
  CMAKE_FLAGS="${CMAKE_FLAGS} -DTHRUST_MULTICONFIG_ENABLE_SYSTEM_CUDA=ON"
  CMAKE_FLAGS="${CMAKE_FLAGS} -DTHRUST_MULTICONFIG_WORKLOAD=LARGE"
else
  # Pre-commit build.
  CMAKE_FLAGS="${CMAKE_FLAGS} -DTHRUST_DISABLE_ARCH_BY_DEFAULT=ON"
  CMAKE_FLAGS="${CMAKE_FLAGS} -DTHRUST_ENABLE_COMPUTE_50=ON"
  CMAKE_FLAGS="${CMAKE_FLAGS} -DTHRUST_ENABLE_COMPUTE_60=ON"
  CMAKE_FLAGS="${CMAKE_FLAGS} -DTHRUST_ENABLE_COMPUTE_70=ON"
  CMAKE_FLAGS="${CMAKE_FLAGS} -DTHRUST_ENABLE_COMPUTE_80=ON"
  CMAKE_FLAGS="${CMAKE_FLAGS} -DTHRUST_INCLUDE_CUB_CMAKE=ON"
  CMAKE_FLAGS="${CMAKE_FLAGS} -DTHRUST_ENABLE_MULTICONFIG=ON"
  CMAKE_FLAGS="${CMAKE_FLAGS} -DTHRUST_MULTICONFIG_ENABLE_DIALECT_CPP11=ON"
  CMAKE_FLAGS="${CMAKE_FLAGS} -DTHRUST_IGNORE_DEPRECATED_CPP_11=ON"
  CMAKE_FLAGS="${CMAKE_FLAGS} -DTHRUST_MULTICONFIG_ENABLE_DIALECT_CPP14=ON"
  CMAKE_FLAGS="${CMAKE_FLAGS} -DTHRUST_MULTICONFIG_ENABLE_DIALECT_CPP17=OFF"
  CMAKE_FLAGS="${CMAKE_FLAGS} -DTHRUST_MULTICONFIG_ENABLE_SYSTEM_CPP=ON"
  CMAKE_FLAGS="${CMAKE_FLAGS} -DTHRUST_MULTICONFIG_ENABLE_SYSTEM_TBB=ON"
  CMAKE_FLAGS="${CMAKE_FLAGS} -DTHRUST_MULTICONFIG_ENABLE_SYSTEM_OMP=ON"
  CMAKE_FLAGS="${CMAKE_FLAGS} -DTHRUST_MULTICONFIG_ENABLE_SYSTEM_CUDA=ON"
  CMAKE_FLAGS="${CMAKE_FLAGS} -DTHRUST_MULTICONFIG_WORKLOAD=SMALL"
fi

CMAKE_BUILD_FLAGS="-j${PARALLEL_LEVEL}"

if [ ! -z "${@}" ]; then
  CMAKE_BUILD_FLAGS="${CMAKE_BUILD_FLAGS} -- ${@}"
fi

CTEST_FLAGS=""

if [ "${BUILD_TYPE}" == "cpu" ]; then
  CTEST_FLAGS="${CTEST_FLAGS} -E ^cub|^thrust.*cuda"
fi

if [ ! -z "${@}" ]; then
  CTEST_FLAGS="${CTEST_FLAGS} -R ^${@}$"
fi

################################################################################
# ENVIRONMENT - Configure and print out information about the environment.
################################################################################

logger "Get environment..."
env

logger "Check versions..."
${CXX} --version
${CUDACXX} --version

################################################################################
# BUILD - Build Thrust and CUB examples and tests.
################################################################################

logger "Configure Thrust and CUB..."
cmake .. ${CMAKE_FLAGS}

logger "Build Thrust and CUB..."
cmake --build . ${CMAKE_BUILD_FLAGS}

################################################################################
# TEST - Run Thrust and CUB examples and tests.
################################################################################

logger "Test Thrust and CUB..."
ctest ${CTEST_FLAGS}

