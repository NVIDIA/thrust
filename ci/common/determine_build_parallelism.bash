#! /usr/bin/env bash

# Copyright (c) 2018-2020 NVIDIA Corporation
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Released under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.

# https://stackoverflow.com/a/23378780
if [ $(uname) == "Darwin" ]; then
  export LOGICAL_CPU_COUNT=$(sysctl -n hw.logicalcpu_max)
  export PHYSICAL_CPU_COUNT=$(sysctl -n hw.physicalcpu_max)
else
  export LOGICAL_CPU_COUNT=$(lscpu -p | egrep -v '^#' | wc -l)
  export PHYSICAL_CPU_COUNT=$(lscpu -p | egrep -v '^#' | sort -u -t, -k 2,4 | wc -l)
fi

export TOTAL_MEM_KB=`grep MemTotal /proc/meminfo | awk '{print $2}'`

export CPU_BOUND_THREADS=$((${PHYSICAL_CPU_COUNT} * 2))           # 2 Build Threads / Core
export MEM_BOUND_THREADS=$((${TOTAL_MEM_KB} / (2 * 1000 * 1000))) # 2 GB / Build Thread

# Pick the smaller of the two as the default.
if [ ${MEM_BOUND_THREADS} -lt ${CPU_BOUND_THREADS} ]; then
  export PARLLEL_LEVEL=${MEM_BOUND_THREADS}
else
  export PARLLEL_LEVEL=${CPU_BOUND_THREADS}
fi

echo "Logical CPU Count:  ${LOGICAL_CPU_COUNT} [threads]"
echo "Physical CPU Count: ${PHYSICAL_CPU_COUNT} [cores]"
echo "Total Mem:          ${TOTAL_MEM_KB} [kb]"
echo "CPU Bound Jobs:     ${CPU_BOUND_THREADS}"
echo "Mem Bound Jobs:     ${MEM_BOUND_THREADS}"
echo "Parallel Level:     ${PARLLEL_LEVEL} [threads]"

