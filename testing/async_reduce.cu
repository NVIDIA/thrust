#include <thrust/detail/config.h>

#if THRUST_CPP_DIALECT >= 2011

#include <unittest/unittest.h>

#include <thrust/async/reduce.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

template <typename T>
struct custom_plus
{
  __host__ __device__
  T operator()(T lhs, T rhs) const
  {
    return lhs + rhs;
  }
};

#define DEFINE_REDUCE_INVOKER(name, ...)                                        \
  template <typename T>                                                       \
  struct name                                                                 \
  {                                                                           \
    template <                                                                \
      typename ForwardIt, typename Sentinel                                   \
    >                                                                         \
    __host__                                                                  \
    static auto sync(                                                         \
      ForwardIt&& first, Sentinel&& last                                      \
    )                                                                         \
    THRUST_DECLTYPE_RETURNS(                                                  \
      ::thrust::reduce(                                                       \
        THRUST_FWD(first), THRUST_FWD(last)                                   \
      )                                                                       \
    )                                                                         \
                                                                              \
    template <                                                                \
      typename ForwardIt, typename Sentinel                                   \
    >                                                                         \
    __host__                                                                  \
    static auto async(                                                        \
      ForwardIt&& first, Sentinel&& last                                      \
    )                                                                         \
    THRUST_DECLTYPE_RETURNS(                                                  \
      ::thrust::async::reduce(                                                \
        __VA_ARGS__                                                           \
        THRUST_PP_COMMA_IF(THRUST_PP_ARITY(__VA_ARGS__))                      \
        THRUST_FWD(first), THRUST_FWD(last)                                   \
      )                                                                       \
    )                                                                         \
  };                                                                          \
  /**/

DEFINE_REDUCE_INVOKER(
  reduce_invoker
);
DEFINE_REDUCE_INVOKER(
  reduce_invoker_device, thrust::device
);

#define DEFINE_REDUCE_INIT_INVOKER(name, init, ...)                           \
  template <typename T>                                                       \
  struct name                                                                 \
  {                                                                           \
    static T call_init() { return init(); }                                   \
                                                                              \
    template <                                                                \
      typename ForwardIt, typename Sentinel                                   \
    >                                                                         \
    __host__                                                                  \
    static auto sync(                                                         \
      ForwardIt&& first, Sentinel&& last                                      \
    )                                                                         \
    THRUST_DECLTYPE_RETURNS(                                                  \
      ::thrust::reduce(                                                       \
        THRUST_FWD(first), THRUST_FWD(last), call_init()                      \
      )                                                                       \
    )                                                                         \
                                                                              \
    template <                                                                \
      typename ForwardIt, typename Sentinel                                   \
    >                                                                         \
    __host__                                                                  \
    static auto async(                                                        \
      ForwardIt&& first, Sentinel&& last                                      \
    )                                                                         \
    THRUST_DECLTYPE_RETURNS(                                                  \
      ::thrust::async::reduce(                                                \
        __VA_ARGS__                                                           \
        THRUST_PP_COMMA_IF(THRUST_PP_ARITY(__VA_ARGS__))                      \
        THRUST_FWD(first), THRUST_FWD(last), call_init()                      \
      )                                                                       \
    )                                                                         \
  };                                                                          \
  /**/

DEFINE_REDUCE_INIT_INVOKER(
  reduce_invoker_init
, [] { return unittest::random_integer<T>(); }
);
DEFINE_REDUCE_INIT_INVOKER(
  reduce_invoker_init_device
, [] { return unittest::random_integer<T>(); }
, thrust::device 
);

#define DEFINE_REDUCE_INIT_OP_INVOKER(name, init, op, ...)                    \
  template <typename T>                                                       \
  struct name                                                                 \
  {                                                                           \
    static T call_init() { return init(); }                                   \
                                                                              \
    template <                                                                \
      typename ForwardIt, typename Sentinel                                   \
    >                                                                         \
    __host__                                                                  \
    static auto sync(                                                         \
      ForwardIt&& first, Sentinel&& last                                      \
    )                                                                         \
    THRUST_DECLTYPE_RETURNS(                                                  \
      ::thrust::reduce(                                                       \
        THRUST_FWD(first), THRUST_FWD(last), call_init(), op<T>{}             \
      )                                                                       \
    )                                                                         \
                                                                              \
    template <                                                                \
      typename ForwardIt, typename Sentinel                                   \
    >                                                                         \
    __host__                                                                  \
    static auto async(                                                        \
      ForwardIt&& first, Sentinel&& last                                      \
    )                                                                         \
    THRUST_DECLTYPE_RETURNS(                                                  \
      ::thrust::async::reduce(                                                \
        __VA_ARGS__                                                           \
        THRUST_PP_COMMA_IF(THRUST_PP_ARITY(__VA_ARGS__))                      \
        THRUST_FWD(first), THRUST_FWD(last), call_init(), op<T>{}             \
      )                                                                       \
    )                                                                         \
  };                                                                          \
  /**/

DEFINE_REDUCE_INIT_OP_INVOKER(
  reduce_invoker_init_plus
, [] { return unittest::random_integer<T>(); }
, thrust::plus
);
DEFINE_REDUCE_INIT_OP_INVOKER(
  reduce_invoker_init_plus_device
, [] { return unittest::random_integer<T>(); }
, thrust::plus
, thrust::device 
);

DEFINE_REDUCE_INIT_OP_INVOKER(
  reduce_invoker_init_custom_plus
, [] { return unittest::random_integer<T>(); }
, custom_plus
);
DEFINE_REDUCE_INIT_OP_INVOKER(
  reduce_invoker_init_custom_plus_device
, [] { return unittest::random_integer<T>(); }
, custom_plus
, thrust::device 
);

#undef DEFINE_REDUCE_INVOKER
#undef DEFINE_REDUCE_INIT_INVOKER
#undef DEFINE_REDUCE_INIT_OP_INVOKER

///////////////////////////////////////////////////////////////////////////////

template <template <typename> class ReduceInvoker>
struct test_async_reduce
{
  template <typename T>
  struct tester
  {
    __host__
    void operator()(std::size_t n)
    {
      thrust::host_vector<T>   h0_data(unittest::random_integers<T>(n));
      thrust::device_vector<T> d0_data(h0_data);

      ASSERT_EQUAL(h0_data, d0_data);

      auto const r0 = ReduceInvoker<T>::sync(
        h0_data.begin(), h0_data.end()
      );

      auto f0 = ReduceInvoker<T>::async(
        d0_data.begin(), d0_data.end()
      );

      auto r1 = f0.consume();

      ASSERT_EQUAL(r0, r1);
    }
  };
};
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES_AND_NAME(
  THRUST_PP_EXPAND_ARGS(
    test_async_reduce<
      reduce_invoker
    >::tester
  )
, NumericTypes
, test_async_reduce
);
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES_AND_NAME(
  THRUST_PP_EXPAND_ARGS(
    test_async_reduce<
      reduce_invoker_device
    >::tester
  )
, NumericTypes
, test_async_reduce_policy
);
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES_AND_NAME(
  THRUST_PP_EXPAND_ARGS(
    test_async_reduce<
      reduce_invoker_init
    >::tester
  )
, NumericTypes
, test_async_reduce_init
);
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES_AND_NAME(
  THRUST_PP_EXPAND_ARGS(
    test_async_reduce<
      reduce_invoker_init_device
    >::tester
  )
, NumericTypes
, test_async_reduce_policy_init
);
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES_AND_NAME(
  THRUST_PP_EXPAND_ARGS(
    test_async_reduce<
      reduce_invoker_init_plus
    >::tester
  )
, NumericTypes
, test_async_reduce_init_plus
);
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES_AND_NAME(
  THRUST_PP_EXPAND_ARGS(
    test_async_reduce<
      reduce_invoker_init_plus_device
    >::tester
  )
, NumericTypes
, test_async_reduce_policy_init_plus
);
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES_AND_NAME(
  THRUST_PP_EXPAND_ARGS(
    test_async_reduce<
      reduce_invoker_init_custom_plus
    >::tester
  )
, NumericTypes
, test_async_reduce_init_custom_plus
);
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES_AND_NAME(
  THRUST_PP_EXPAND_ARGS(
    test_async_reduce<
      reduce_invoker_init_custom_plus_device
    >::tester
  )
, NumericTypes
, test_async_reduce_policy_init_custom_plus
);

///////////////////////////////////////////////////////////////////////////////

// TODO: counting_iterator.

// TODO: Async copy then reduce.

// TODO: Device-side reduction usage.

// TODO: Make random_integers more generic.

#endif // THRUST_CPP_DIALECT >= 2011

