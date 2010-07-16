#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/reduce.h>
#include <thrust/iterator/detail/placement/is_placed.h>
#include <thrust/iterator/detail/placement/place.h>
#include <cassert>

inline cudaError_t cudaThreadExitActiveDevice(void)
{
    int oldDevice = -1;
    cudaError_t error = cudaSuccess;

    // get the old device index
    error = cudaGetDevice(&oldDevice);
    if (cudaSuccess != error) {
        return error;
    }

    // exit old device's context
    error = cudaThreadExit();
    if (cudaSuccess != error) {
        return error;
    }
    thrust::detail::place_detail::contexts[oldDevice] = NULL;

    return cudaSuccess;
}

int reduce(const thrust::host_vector<thrust::device_vector<int> > &vectors)
{
  int result = 0;

  for(int p = 0; p < vectors.size(); ++p)
  {
    thrust::detail::push_place(p);
    result += thrust::reduce(vectors[p].begin(), vectors[p].end());
    thrust::detail::pop_place();
  }

  return result;
}

int main(void)
{
  int N = 1000;

  int num_places = thrust::detail::num_places();

  thrust::host_vector<thrust::device_vector<int> > vectors(num_places);

  for(int p = 0; p < num_places; ++p)
  {
    thrust::detail::push_place(p);
    vectors[p].resize(N);
    thrust::fill(vectors[p].begin(), vectors[p].end(), 13);
    thrust::detail::pop_place();
  }

  assert(reduce(vectors) == 13 * N * num_places);

  for(int p = 0; p < num_places; ++p)
  {
    thrust::detail::push_place(p);

    // deallocate the vector while we're in the right place
    vectors[p].resize(0);
    vectors[p].shrink_to_fit();
    cudaThreadExitActiveDevice();
    thrust::detail::pop_place();
  }

  return 0;
}

