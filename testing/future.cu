#include <thrust/detail/config.h>

#if THRUST_CPP_DIALECT >= 2011

#include <unittest/unittest.h>

#include <thrust/future.h>

struct mock {};

using future_non_void_value_types = unittest::type_list<
  char
, signed char
, unsigned char
, short
, unsigned short
, int
, unsigned int
, long
, unsigned long
, long long
, unsigned long long
, float
, double
, custom_numeric
, float2
, mock
>;

using future_value_types = unittest::type_list<
  char
, signed char
, unsigned char
, short
, unsigned short
, int
, unsigned int
, long
, unsigned long
, long long
, unsigned long long
, float
, double
, custom_numeric
, float2
, mock
, void
>;

///////////////////////////////////////////////////////////////////////////////

template <typename T>
struct test_future_default_constructed
{
  template <typename Future>
  __host__
  static void per_future(Future&& f)
  {
    ASSERT_EQUAL(false, f.valid());

    ASSERT_THROWS_EQUAL(
      f.wait()
    , thrust::future_error
    , thrust::future_error(thrust::future_errc::no_state)
    );

    ASSERT_THROWS_EQUAL(
      f.stream()
    , thrust::future_error
    , thrust::future_error(thrust::future_errc::no_state)
    );
  }

  __host__
  void operator()()
  {
    thrust::future<T>                                  f0;
    thrust::future<T, decltype(thrust::device)>        f1;
    thrust::future<T, decltype(thrust::cuda::par)>     f2;
    thrust::future<T, decltype(thrust::device),    T*> f3;
    thrust::future<T, decltype(thrust::cuda::par), T*> f4;

    per_future(f0);
    per_future(f1);
    per_future(f2);
    per_future(f3);
    per_future(f4);
  }
};
DECLARE_GENERIC_UNITTEST_WITH_TYPES(
  test_future_default_constructed
, future_value_types
);

///////////////////////////////////////////////////////////////////////////////

// TODO: CUDA specific tests, e.g. where(), stream callbacks
 
#endif // THRUST_CPP_DIALECT >= 2011

