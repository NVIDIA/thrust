#!/bin/bash

THRUST="`dirname \"$0\"`"             # relative
THRUST="`( cd \"$THRUST\"/.. && pwd )`"  # absolutized and normalized

THRUST_BUILD=$THRUST/build
THRUST_EXAMPLES=$THRUST/examples
THRUST_TESTING=$THRUST/testing
THRUST_TRIVIAL_TESTS=$THRUST/testing/trivial_tests

cpu_host_systems="cpp omp tbb" 
cpu_device_systems="omp tbb"
device_systems="cuda omp tbb"

num_gpus=`ls /proc/driver/nvidia/gpus/ | wc -w`

# build the full product of (host_backend,backend) for examples and trivial tests
for host in $cpu_host_systems; do
    for device in $cpu_device_systems; do
        cd $THRUST_EXAMPLES
        scons -j2 host_backend=$host backend=$device run_examples
        cd $THRUST_TRIVIAL_TESTS
        scons -j2 host_backend=$host backend=$device
        ./tester
    done
done

if (( num_gpus > 0)); then
  nvcc $THRUST_BUILD/print_sm_version.cpp -o /tmp/print_sm_version
  for host in $cpu_host_systems; do
      for ((device_id=0;device_id<num_gpus;device_id++)); do
          arch=`/tmp/print_sm_version $device_id`
          cd $THRUST_EXAMPLES
          scons -j2 host_backend=$host backend=cuda arch=$arch run_examples
          cd $THRUST_TRIVIAL_TESTS
          scons -j2 host_backend=$host backend=$device
          ./tester
      done
  done
fi


# build unit tests for each cpu device systems
for device in $cpu_device_systems; do
    cd $THRUST_TESTING
    scons -j2 host_backend=cpp backend=$device
    ./tester
done

if (( num_gpus > 0)); then
  nvcc $THRUST_BUILD/print_sm_version.cpp -o /tmp/print_sm_version
  for ((device_id=0;device_id<num_gpus;device_id++)); do
      arch=`/tmp/print_sm_version $device_id`
      cd $THRUST_TESTING
      scons -j2 host_backend=cpp backend=cuda
      ./tester
  done
fi

