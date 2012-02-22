#!/bin/bash

THRUST="`dirname \"$0\"`"             # relative
THRUST="`( cd \"$THRUST\"/.. && pwd )`"  # absolutized and normalized

THRUST_BUILD=$THRUST/build
THRUST_EXAMPLES=$THRUST/examples
THRUST_TESTING=$THRUST/testing

cpu_host_systems="cpp omp tbb" 
cpu_device_systems="omp tbb"

for host in $cpu_host_systems; do
    for device in $cpu_device_systems; do
        cd $THRUST_EXAMPLES
        scons -j2 host_backend=$host backend=$device run_examples
        cd $THRUST_TESTING
        scons -j2 host_backend=$host backend=$device
        ./tester
    done
done

num_gpus=`ls /proc/driver/nvidia/gpus/ | wc -w`

if (( num_gpus > 0)); then
  nvcc $THRUST_BUILD/print_sm_version.cpp -o /tmp/print_sm_version
  for host in $cpu_host_systems; do
      for ((device_id=0;device_id<num_gpus;device_id++)); do
          arch=`/tmp/print_sm_version $device_id`
          cd $THRUST_EXAMPLES
          scons -j2 host_backend=$host backend=cuda arch=$arch run_examples
          cd $THRUST_TESTING
          scons -j2 host_backend=$host backend=cuda arch=$arch
          ./tester --device=$device_id
      done
  done
fi

