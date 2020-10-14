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

# Set path and build parallel level
export PATH=/usr/local/cuda/bin:${PATH}

# Set home to the job's workspace.
export HOME=${WORKSPACE}

# Switch to project root; also root of repo checkout.
cd ${WORKSPACE}

# If it's a nightly build, append current YYMMDD to version.
if [[ "${BUILD_MODE}" = "branch" ]]; then
  export VERSION_SUFFIX=`date +%y%m%d`
fi

# The Docker image sets up `c++` and `cu++`.
CMAKE_FLAGS="-DCMAKE_CXX_COMPILER=c++ -DCMAKE_CUDA_COMPILER=cu++"

# If it's a nightly build, build all configurations.
if [[ "${BUILD_MODE}" = "branch" ]]; then
  CMAKE_FLAGS="${CMAKE_FLAGS} -DTHRUST_MULTICONFIG_WORKLOAD=FULL"
fi

CTEST_FLAGS=""

if [[ "${BUILD_KIND}" = "cpu" ]]; then
  CTEST_FLAGS="${CTEST_FLAGS} -E '^cub|^thrust.*cuda'"
fi

################################################################################
# ENVIRONMENT - Print out information about the environment.
################################################################################

logger "Get environment..."
env

logger "Check versions..."
c++ --version
cu++ --version

################################################################################
# BUILD - Build Thrust and CUB examples and tests.
################################################################################

mkdir -p build
cd build

logger "Configure Thrust and CUB..."
cmake ${CMAKE_FLAGS} ..

logger "Build Thrust and CUB..."
cmake --build . -j "${1}"

################################################################################
# TEST - Run Thrust and CUB examples and tests.
################################################################################

logger "Test Thrust and CUB..."
ctest ${CTEST_FLAGS}

