#! /usr/bin/env bash

# Copyright (c) 2018-2020 NVIDIA Corporation
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Released under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.

function usage {
  echo "Usage: ${0} [flags...]"
  echo
  echo "Examine the system topology to determine a reasonable amount of build"
  echo "parallelism."
  echo
  echo "Exported variables:"
  echo "  $${LOGICAL_CPUS}      : Logical processors (e.g. hyperthreads)."
  echo "  $${PHYSICAL_CPUS}     : Physical processors (e.g. cores)."
  echo "  $${TOTAL_MEM_KB}      : Total system memory."
  echo "  $${CPU_BOUND_THREADS} : # of build threads constrained by processors."
  echo "  $${MEM_BOUND_THREADS} : # of build threads constrained by memory."
  echo "  $${PARLLEL_LEVEL}     : Determined # of build threads."
  echo
  echo "-h, -help, --help"
  echo "  Print this message."
  echo
  echo "-q, --quiet"
  echo "  Print nothing and only export variables."

  exit -3
}

QUIET=0

while test ${#} != 0
do
  case "${1}" in
  -h) ;&
  -help) ;&
  --help) usage ;;
  -q) ;&
  --quiet) QUIET=1 ;;
  esac
  shift
done

# https://stackoverflow.com/a/23378780
if [ $(uname) == "Darwin" ]; then
  export LOGICAL_CPUS=$(sysctl -n hw.logicalcpu_max)
  export PHYSICAL_CPUS=$(sysctl -n hw.physicalcpu_max)
else
  export LOGICAL_CPUS=$(lscpu -p | egrep -v '^#' | wc -l)
  export PHYSICAL_CPUS=$(lscpu -p | egrep -v '^#' | sort -u -t, -k 2,4 | wc -l)
fi

export TOTAL_MEM_KB=`grep MemTotal /proc/meminfo | awk '{print $2}'`

export CPU_BOUND_THREADS=$((${PHYSICAL_CPUS} * 2))                # 2 Build Threads / Core
export MEM_BOUND_THREADS=$((${TOTAL_MEM_KB} / (2 * 1000 * 1000))) # 2 GB / Build Thread

# Pick the smaller of the two as the default.
if [ ${MEM_BOUND_THREADS} -lt ${CPU_BOUND_THREADS} ]; then
  export PARLLEL_LEVEL=${MEM_BOUND_THREADS}
else
  export PARLLEL_LEVEL=${CPU_BOUND_THREADS}
fi

if [ "${QUIET}" == 0 ]; then
  echo "Logical CPUs:      ${LOGICAL_CPUS} [threads]"
  echo "Physical CPUs:     ${PHYSICAL_CPUS} [cores]"
  echo "Total Mem:         ${TOTAL_MEM_KB} [kb]"
  echo "CPU Bound Threads: ${CPU_BOUND_THREADS} [threads]"
  echo "Mem Bound Threads: ${MEM_BOUND_THREADS} [threads]"
  echo "Parallel Level:    ${PARLLEL_LEVEL} [threads]"
fi

