#include <thrust/device_vector.h>

#ifdef DIRECT_CALL // reduce directly uses THRUST_CNP_DISPATCH, etc
#include <thrust/reduce.h>
#else // equal indirectly uses THRUST_CNP_DISPATCH, etc
#include <thrust/equal.h>
#endif

#include <iostream>

template <typename ExecutionPolicy,
          typename Iterator,
          typename T,
          typename OutputIter>
__global__
void test_kernel(ExecutionPolicy exec,
                   Iterator first,
                   Iterator last,
                   T init,
                   OutputIter out)
{
#ifdef DIRECT_CALL
  *out = thrust::reduce(exec, first, last, init);
#else
  (void)init;
  *out = thrust::equal(exec, first, last, first);
#endif
}

int main()
{
  thrust::device_vector<int> vec(21, 1);
  thrust::device_vector<bool> equal_out(1, false);
  test_kernel()<<<1, 1>>>(EXEC_POLICY,
                          vec.cbegin() + 1,
                          vec.cend(),
                          3,
#ifdef DIRECT_CALL
                          vec.begin()
#else
                          equal_out.begin()
#endif
                          );

  if (CubDebug(cudaDeviceSynchronize()))
  {
    std::cerr << "CUDA error encountered.\n";
    return 1;
  }

#ifdef DIRECT_CALL
  const int result = vec[0];
  const int expected = 23;
#else
  const bool result = equal_out[0];
  const bool expected = true;
#endif

  std::cout << "Result: " << result << std::endl;

  if (result != expected)
  {
    std::cerr << "Expected '" << expected << "'.\n";
    return 1;
  }

  return 0;
}
