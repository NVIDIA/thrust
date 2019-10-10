// Regression test for NVBug 2720132.

#include <unittest/unittest.h>
#include <thrust/device_vector.h>
#include <thrust/detail/cstdint.h>

struct non_trivial
{
  __host__ __device__ non_trivial() {}
  __host__ __device__ ~non_trivial() {}
};

void test_out_of_memory_recovery()
{
  try
  {
    thrust::device_vector<non_trivial> x(1);

    for (thrust::detail::uint64_t n = 1 ;; n <<= 1)
      thrust::device_vector<thrust::detail::uint32_t> y(n);
  }
  catch (...) { }
}
DECLARE_UNITTEST(test_out_of_memory_recovery);
