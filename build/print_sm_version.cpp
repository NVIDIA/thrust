#include <cuda_runtime_api.h>
#include <stdio.h>

int main(void)
{
  int num_devices = 0;
  cudaGetDeviceCount(&num_devices);
  if(num_devices > 0)
  {
    cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties, 0);
    printf("--gpu-architecture=sm_%d%d", properties.major, properties.minor);
    return 0;
  } // end if

  return -1;
}

