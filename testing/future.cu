#include <thrust/detail/config.h>

#if THRUST_CPP_DIALECT >= 2011

#include <unittest/unittest.h>

#include <thrust/future.h>

template <typename T>
void test_future_default_construction()
{
  thrust::future<T>                                  f0;
  thrust::future<T, decltype(thrust::device)>        f1;
  thrust::future<T, decltype(thrust::cuda::par)>     f2;
  thrust::future<T, decltype(thrust::device),    T*> f3;
  thrust::future<T, decltype(thrust::cuda::par), T*> f4;

  ASSERT_EQUAL(false, f0.valid());
  ASSERT_EQUAL(false, f1.valid());
  ASSERT_EQUAL(false, f2.valid());
  ASSERT_EQUAL(false, f3.valid());
  ASSERT_EQUAL(false, f4.valid());
};
DECLARE_GENERIC_UNITTEST(test_future_default_construction);

#endif // THRUST_CPP_DIALECT >= 2011

