#include <thrust/detail/config.h>

#if THRUST_CPP_DIALECT >= 2011

#include <unittest/unittest.h>

#include <thrust/async/transform.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#define DEFINE_ASYNC_TRANSFORM_CALLABLE(name, ...)                            \
  struct THRUST_PP_CAT2(name, _fn)                                            \
  {                                                                           \
    template <                                                                \
      typename ForwardIt, typename Sentinel, typename OutputIt                \
    , typename UnaryOperation                                                 \
    >                                                                         \
    __host__                                                                  \
    auto operator()(                                                          \
      ForwardIt&& first, Sentinel&& last, OutputIt&& output                   \
    , UnaryOperation&& f                                                      \
    ) const                                                                   \
    THRUST_DECLTYPE_RETURNS(                                                  \
      ::thrust::async::transform(                                             \
        __VA_ARGS__                                                           \
        THRUST_PP_COMMA_IF(THRUST_PP_ARITY(__VA_ARGS__))                      \
        THRUST_FWD(first), THRUST_FWD(last), THRUST_FWD(output), THRUST_FWD(f)\
      )                                                                       \
    )                                                                         \
  };                                                                          \
  /**/

DEFINE_ASYNC_TRANSFORM_CALLABLE(
  invoke_async_transform
);

DEFINE_ASYNC_TRANSFORM_CALLABLE(
  invoke_async_transform_device, thrust::device
);

#undef DEFINE_ASYNC_TRANSFORM_CALLABLE

///////////////////////////////////////////////////////////////////////////////

struct divide_by_2
{
  template <typename T>
  __host__ __device__
  T operator()(T x) const
  {
    return x / 2;
  }
};

///////////////////////////////////////////////////////////////////////////////

template <typename AsyncTransformCallable, typename UnaryOperation>
struct test_async_transform_unary
{
  template <typename T>
  struct tester
  {
    __host__
    void operator()(std::size_t n)
    {
      thrust::host_vector<T>   h0_data(unittest::random_integers<T>(n));
      thrust::device_vector<T> d0_data(h0_data);

      thrust::host_vector<T>   h1_data(n);
      thrust::device_vector<T> d1_data(n);

      thrust::transform(
        h0_data.begin(), h0_data.end(), h1_data.begin(), UnaryOperation{}
      );

      auto f0 = AsyncTransformCallable{}(
        d0_data.begin(), d0_data.end(), d1_data.begin(), UnaryOperation{}
      );

      f0.wait();

      ASSERT_EQUAL(h0_data, d0_data);
      ASSERT_EQUAL(h1_data, d1_data);
    }
  };
};
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES_AND_NAME(
  THRUST_PP_EXPAND_ARGS(
    test_async_transform_unary<
      invoke_async_transform_fn
    , divide_by_2
    >::tester
  )
, NumericTypes
, test_async_transform_unary
);
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES_AND_NAME(
  THRUST_PP_EXPAND_ARGS(
    test_async_transform_unary<
      invoke_async_transform_device_fn
    , divide_by_2
    >::tester
  )
, NumericTypes
, test_async_transform_unary_policy
);

///////////////////////////////////////////////////////////////////////////////

template <typename AsyncTransformCallable, typename UnaryOperation>
struct test_async_transform_unary_inplace
{
  template <typename T>
  struct tester
  {
    __host__
    void operator()(std::size_t n)
    {
      thrust::host_vector<T>   h0_data(unittest::random_integers<T>(n));
      thrust::device_vector<T> d0_data(h0_data);

      thrust::transform(
        h0_data.begin(), h0_data.end(), h0_data.begin(), UnaryOperation{}
      );

      auto f0 = AsyncTransformCallable{}(
        d0_data.begin(), d0_data.end(), d0_data.begin(), UnaryOperation{}
      );

      f0.wait();

      ASSERT_EQUAL(h0_data, d0_data);
    }
  };
};
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES_AND_NAME(
  THRUST_PP_EXPAND_ARGS(
    test_async_transform_unary_inplace<
      invoke_async_transform_fn
    , divide_by_2
    >::tester
  )
, NumericTypes
, test_async_transform_unary_inplace
);
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES_AND_NAME(
  THRUST_PP_EXPAND_ARGS(
    test_async_transform_unary_inplace<
      invoke_async_transform_device_fn
    , divide_by_2
    >::tester
  )
, NumericTypes
, test_async_transform_unary_inplace_policy
);

#endif // THRUST_CPP_DIALECT >= 2011

