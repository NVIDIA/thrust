#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>

void usage(const char *name)
{
  printf("usage: %s [device_id]\n", name);
}

int main(int argc, char **argv)
{
  int num_devices = 0;
  int device_id = 0;
  if(argc == 2)
  {
    device_id = atoi(argv[1]);
  }
  else if(argc > 2)
  {
    usage(argv[0]);
    exit(-1);
  }

  cudaGetDeviceCount(&num_devices);
  if(num_devices > device_id)
  {
    cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties, device_id);
    printf("--gpu-architecture=sm_%d%d", properties.major, properties.minor);
    return 0;
  } // end if
  else
  {
    printf("No available device with id %d\n", device_id);
  }

  return -1;
}

