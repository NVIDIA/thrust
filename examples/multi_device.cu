#include <cassert>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/reduce.h>
#include "cuda_multidev.h"

int reduce(const thrust::host_vector<thrust::device_vector<int> > &vectors)
{
  int result = 0;

  for(int d = 0; d < vectors.size(); ++d)
  {
    cudaSetActiveDevice(d);
    result += thrust::reduce(vectors[d].begin(), vectors[d].end());
  }

  return result;
}

int main(void)
{
  int num_devices = 0;
  int N = 1000;

  cudaGetDeviceCount(&num_devices);

  thrust::host_vector<thrust::device_vector<int> > vectors(num_devices);

  for(int d = 0; d < num_devices; ++d)
  {
    cudaSetActiveDevice(d);
    vectors[d].resize(N);
    thrust::fill(vectors[d].begin(), vectors[d].end(), 13);
  }

  assert(reduce(vectors) == 13 * N * num_devices);

  // explicitly dealloc so that we free on the right device upon teardown
  for(int d = 0; d < num_devices; ++d)
  {
    cudaSetActiveDevice(d);
    vectors[d].clear();
    vectors[d].shrink_to_fit();
    cudaThreadExitActiveDevice();
  }

  return 0;
}

