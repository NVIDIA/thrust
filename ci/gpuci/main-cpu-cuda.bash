#! /usr/bin/env bash

# Copyright (c) 2018-2022 NVIDIA Corporation
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Released under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.

# Parse this script name to get the relevant info:
export BUILD_NAME=$(basename "${BASH_SOURCE}" .bash)

# Call the common routine:
source ${WORKSPACE}/ci/gpuci/gpuci_common_config.bash
