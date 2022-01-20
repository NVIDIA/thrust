#! /usr/bin/env bash

# Copyright (c) 2018-2022 NVIDIA Corporation
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Released under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.

# Define BUILD_NAME to the name of the per-config bash script:
# `export BUILD_NAME=$(basename "${BASH_SOURCE}" .bash)`
# before entering this file.

BUILD_REGEX="^([^-]+)-([^-]+)-([^-]+)$"

if [[ ${BUILD_NAME} =~ ${BUILD_REGEX} ]]; then
  export BUILD_TRIGGER=${BASH_REMATCH[1]}
  export BUILD_TYPE=${BASH_REMATCH[2]}
  export BUILD_SUBSET=${BASH_REMATCH[3]}
else
  echo "Unexpected build name: '${BUILD_NAME}'"
  exit 1
fi

export PARALLEL_LEVEL=${PARALLEL_LEVEL:-4}

export CMAKE_CONFIG_PRESET="gpuci-${BUILD_NAME}"
export CMAKE_TEST_PRESET="gpuci-${BUILD_NAME}"

source ${WORKSPACE}/ci/common/build.bash
