#include <thrust/detail/config.h>

// Disabled on MSVC for GH issue #1098
#if THRUST_CPP_DIALECT >= 2011 && !defined(THRUST_LEGACY_GCC) && \
  THRUST_HOST_COMPILER != THRUST_HOST_COMPILER_MSVC

#include <unittest/unittest.h>

#include <thrust/async/sort.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

enum wait_policy
{
  wait_for_events
, do_not_wait_for_events
};

template <typename T>
struct custom_greater
{
  __host__ __device__
  bool operator()(T rhs, T lhs) const
  {
    return lhs > rhs;
  }
};

#define DEFINE_SORT_INVOKER(name, ...)                                        \
  template <typename T>                                                       \
  struct name                                                                 \
  {                                                                           \
    template <                                                                \
      typename ForwardIt, typename Sentinel                                   \
    >                                                                         \
    __host__                                                                  \
    static void sync(                                                         \
      ForwardIt&& first, Sentinel&& last                                      \
    )                                                                         \
    {                                                                         \
      ::thrust::sort(                                                         \
        THRUST_FWD(first), THRUST_FWD(last)                                   \
      );                                                                      \
    }                                                                         \
                                                                              \
    template <                                                                \
      typename ForwardIt, typename Sentinel                                   \
    >                                                                         \
    __host__                                                                  \
    static auto async(                                                        \
      ForwardIt&& first, Sentinel&& last                                      \
    )                                                                         \
    THRUST_DECLTYPE_RETURNS(                                                  \
      ::thrust::async::sort(                                                  \
        __VA_ARGS__                                                           \
        THRUST_PP_COMMA_IF(THRUST_PP_ARITY(__VA_ARGS__))                      \
        THRUST_FWD(first), THRUST_FWD(last)                                   \
      )                                                                       \
    )                                                                         \
  };                                                                          \
  /**/

DEFINE_SORT_INVOKER(
  sort_invoker
);
DEFINE_SORT_INVOKER(
  sort_invoker_device, thrust::device
);

#define DEFINE_SORT_OP_INVOKER(name, op, ...)                                 \
  template <typename T>                                                       \
  struct name                                                                 \
  {                                                                           \
    template <                                                                \
      typename ForwardIt, typename Sentinel                                   \
    >                                                                         \
    __host__                                                                  \
    static void sync(                                                         \
      ForwardIt&& first, Sentinel&& last                                      \
    )                                                                         \
    {                                                                         \
      ::thrust::sort(                                                         \
        THRUST_FWD(first), THRUST_FWD(last), op<T>{}                          \
      );                                                                      \
    }                                                                         \
                                                                              \
    template <                                                                \
      typename ForwardIt, typename Sentinel                                   \
    >                                                                         \
    __host__                                                                  \
    static auto async(                                                        \
      ForwardIt&& first, Sentinel&& last                                      \
    )                                                                         \
    THRUST_DECLTYPE_RETURNS(                                                  \
      ::thrust::async::sort(                                                  \
        __VA_ARGS__                                                           \
        THRUST_PP_COMMA_IF(THRUST_PP_ARITY(__VA_ARGS__))                      \
        THRUST_FWD(first), THRUST_FWD(last), op<T>{}                          \
      )                                                                       \
    )                                                                         \
  };                                                                          \
  /**/

DEFINE_SORT_OP_INVOKER(
  sort_invoker_less,        thrust::less
);
DEFINE_SORT_OP_INVOKER(
  sort_invoker_less_device, thrust::less, thrust::device
);

DEFINE_SORT_OP_INVOKER(
  sort_invoker_greater,        thrust::greater
);
DEFINE_SORT_OP_INVOKER(
  sort_invoker_greater_device, thrust::greater, thrust::device
);

DEFINE_SORT_OP_INVOKER(
  sort_invoker_custom_greater,        custom_greater
);
DEFINE_SORT_OP_INVOKER(
  sort_invoker_custom_greater_device, custom_greater, thrust::device
);

#undef DEFINE_SORT_INVOKER
#undef DEFINE_SORT_OP_INVOKER

///////////////////////////////////////////////////////////////////////////////

template <template <typename> class SortInvoker, wait_policy WaitPolicy>
struct test_async_sort
{
  template <typename T>
  struct tester
  {
    __host__
    void operator()(std::size_t n)
    {
      thrust::host_vector<T>   h0(unittest::random_integers<T>(n));
      thrust::device_vector<T> d0(h0);

      ASSERT_EQUAL(h0, d0);

      SortInvoker<T>::sync(
        h0.begin(), h0.end()
      );

      auto e0 = SortInvoker<T>::async(
        d0.begin(), d0.end()
      );

      if (wait_for_events == WaitPolicy)
      {
        TEST_EVENT_WAIT(e0);

        ASSERT_EQUAL(h0, d0);
      }
    }
  };
};
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES_AND_NAME(
  THRUST_PP_EXPAND_ARGS(
    test_async_sort<
      sort_invoker
    , wait_for_events
    >::tester
  )
, NumericTypes
, test_async_sort
);
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES_AND_NAME(
  THRUST_PP_EXPAND_ARGS(
    test_async_sort<
      sort_invoker
    , do_not_wait_for_events
    >::tester
  )
, NumericTypes
, test_async_sort_no_wait
);
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES_AND_NAME(
  THRUST_PP_EXPAND_ARGS(
    test_async_sort<
      sort_invoker_device
    , wait_for_events
    >::tester
  )
, NumericTypes
, test_async_sort_policy
);
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES_AND_NAME(
  THRUST_PP_EXPAND_ARGS(
    test_async_sort<
      sort_invoker_device
    , do_not_wait_for_events
    >::tester
  )
, NumericTypes
, test_async_sort_policy_no_wait
);
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES_AND_NAME(
  THRUST_PP_EXPAND_ARGS(
    test_async_sort<
      sort_invoker_less
    , wait_for_events
    >::tester
  )
, NumericTypes
, test_async_sort_less
);
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES_AND_NAME(
  THRUST_PP_EXPAND_ARGS(
    test_async_sort<
      sort_invoker_less
    , do_not_wait_for_events
    >::tester
  )
, NumericTypes
, test_async_sort_less_no_wait
);
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES_AND_NAME(
  THRUST_PP_EXPAND_ARGS(
    test_async_sort<
      sort_invoker_less_device
    , wait_for_events
    >::tester
  )
, NumericTypes
, test_async_sort_policy_less
);
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES_AND_NAME(
  THRUST_PP_EXPAND_ARGS(
    test_async_sort<
      sort_invoker_less_device
    , do_not_wait_for_events
    >::tester
  )
, NumericTypes
, test_async_sort_policy_less_no_wait
);
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES_AND_NAME(
  THRUST_PP_EXPAND_ARGS(
    test_async_sort<
      sort_invoker_greater
    , wait_for_events
    >::tester
  )
, NumericTypes
, test_async_sort_greater
);
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES_AND_NAME(
  THRUST_PP_EXPAND_ARGS(
    test_async_sort<
      sort_invoker_greater
    , do_not_wait_for_events
    >::tester
  )
, NumericTypes
, test_async_sort_greater_no_wait
);
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES_AND_NAME(
  THRUST_PP_EXPAND_ARGS(
    test_async_sort<
      sort_invoker_greater_device
    , wait_for_events
    >::tester
  )
, NumericTypes
, test_async_sort_policy_greater
);
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES_AND_NAME(
  THRUST_PP_EXPAND_ARGS(
    test_async_sort<
      sort_invoker_greater_device
    , do_not_wait_for_events
    >::tester
  )
, NumericTypes
, test_async_sort_policy_greater_no_wait
);
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES_AND_NAME(
  THRUST_PP_EXPAND_ARGS(
    test_async_sort<
      sort_invoker_custom_greater
    , wait_for_events
    >::tester
  )
, NumericTypes
, test_async_sort_custom_greater
);
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES_AND_NAME(
  THRUST_PP_EXPAND_ARGS(
    test_async_sort<
      sort_invoker_custom_greater
    , do_not_wait_for_events
    >::tester
  )
, NumericTypes
, test_async_sort_custom_greater_no_wait
);
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES_AND_NAME(
  THRUST_PP_EXPAND_ARGS(
    test_async_sort<
      sort_invoker_custom_greater_device
    , wait_for_events
    >::tester
  )
, NumericTypes
, test_async_sort_policy_custom_greater
);
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES_AND_NAME(
  THRUST_PP_EXPAND_ARGS(
    test_async_sort<
      sort_invoker_custom_greater_device
    , do_not_wait_for_events
    >::tester
  )
, NumericTypes
, test_async_sort_policy_custom_greater_no_wait
);

///////////////////////////////////////////////////////////////////////////////

template <typename T>
struct test_async_sort_after
{
  __host__
  void operator()(std::size_t n)
  {
    thrust::host_vector<T>   h0(unittest::random_integers<T>(n));
    thrust::device_vector<T> d0(h0);
    thrust::device_vector<T> d1(h0);

    ASSERT_EQUAL(h0, d0);
    ASSERT_EQUAL(h0, d1);

    auto e0 = thrust::async::sort(
      d0.begin(), d0.end()
    );

    ASSERT_EQUAL(true, e0.valid_stream());

    auto const e0_stream = e0.stream().native_handle();

    auto e1 = thrust::async::sort(
      thrust::device.after(e0), d0.begin(), d0.end()
    );

    // Verify that double consumption of a policy produces an exception.
    ASSERT_THROWS_EQUAL(
      auto x = thrust::async::sort(
        thrust::device.after(e0), d0.begin(), d0.end()
      );
      THRUST_UNUSED_VAR(x)
    , thrust::event_error
    , thrust::event_error(thrust::event_errc::no_state)
    );

    ASSERT_EQUAL_QUIET(e0_stream, e1.stream().native_handle());

    auto after_policy2 = thrust::device.after(e1);

    auto e2 = thrust::async::sort(
      after_policy2, d1.begin(), d1.end()
    );

    // Verify that double consumption of a policy produces an exception.
    ASSERT_THROWS_EQUAL(
      auto x = thrust::async::sort(
        after_policy2, d1.begin(), d1.end()
      );
      THRUST_UNUSED_VAR(x)
    , thrust::event_error
    , thrust::event_error(thrust::event_errc::no_state)
    );

    ASSERT_EQUAL_QUIET(e0_stream, e2.stream().native_handle());

    TEST_EVENT_WAIT(e2);

    thrust::sort(h0.begin(), h0.end());

    ASSERT_EQUAL(h0, d0);
    ASSERT_EQUAL(h0, d1);
  }
};
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES(
  test_async_sort_after
, NumericTypes
);

///////////////////////////////////////////////////////////////////////////////

// TODO: Async copy then sort.

// TODO: Test future return type.

#endif

