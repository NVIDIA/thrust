#!/usr/bin/env bash
# Copyright (c) 2018-2020 NVIDIA Corporation

#################################
# Thrust CPU-only script for CI #
#################################

set -e

# Logger function for build status output
function logger() {
  echo -e "\n>>>> ${@}\n"
}

# Set path and build parallel level
export PATH=/usr/local/cuda/bin:${PATH}

# Set home to the job's workspace.
export HOME=${WORKSPACE}

# Switch to project root; also root of repo checkout.
cd ${WORKSPACE}

# If it's a nightly build, append current YYMMDD to version.
if [[ "${BUILD_MODE}" = "branch" ]] ; then
  export VERSION_SUFFIX=`date +%y%m%d`
fi

# The Docker image sets up `c++` and `cu++`.
CMAKE_FLAGS="-DCMAKE_CXX_COMPILER=c++ -DCMAKE_CUDA_COMPILER=cu++"

# If it's a nightly build, build all configurations.
if [[ "${BUILD_MODE}" = "branch" ]] ; then
  CMAKE_FLAGS="${CMAKE_FLAGS} -DTHRUST_MULTICONFIG_WORKLOAD=FULL"
fi

################################################################################
# SETUP - Check environment.
################################################################################

logger "Get env..."
env

logger "Check versions..."
c++ --version
cu++ --version

################################################################################
# BUILD - Build Thrust examples and tests.
################################################################################

mkdir build
cd build

logger "Configure Thrust..."
cmake ${CMAKE_OPTIONS} ..

logger "Build Thrust..."
cmake --build . -j

################################################################################
# TEST - Run Thrust CPU-only examples and tests.
################################################################################

logger "Test Thrust (CPU-only)..."
ctest -E "^cub|^thrust.*cuda"

