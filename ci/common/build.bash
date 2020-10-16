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

# Switch to project root; also root of repo checkout.
cd ${WORKSPACE}

# If it's a nightly build, append current YYMMDD to version.
if [ "${BUILD_MODE}" == "branch" ]; then
  export VERSION_SUFFIX=`date +%y%m%d`
fi

# The Docker image sets up `${CXX}` and `${CUDACXX}`.
CMAKE_FLAGS="-DCMAKE_CXX_COMPILER=${CXX} -DCMAKE_CUDA_COMPILER=${CUDACXX}"

# If it's a nightly build, build all configurations.
if [ "${BUILD_MODE}" == "branch" ]; then
  CMAKE_FLAGS="${CMAKE_FLAGS} -DTHRUST_MULTICONFIG_WORKLOAD=FULL"
fi

CTEST_FLAGS=""

if [ "${BUILD_KIND}" == "cpu" ]; then
  CTEST_FLAGS="${CTEST_FLAGS} -E '^cub|^thrust.*cuda'"
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

mkdir -p build
cd build

if [ ! -f CMakeLists.txt ]; then
  logger "Configure Thrust and CUB..."
  cmake ${CMAKE_FLAGS} ..
else
  logger "Existing Thrust and CUB configuration found, skipping configure..."
fi

logger "Build Thrust and CUB..."
cmake --build . -j${PARALLEL_LEVEL} "${@}"

################################################################################
# TEST - Run Thrust and CUB examples and tests.
################################################################################

logger "Test Thrust and CUB..."
ctest ${CTEST_FLAGS}

