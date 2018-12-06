#include <thrust/detail/config.h>

#if THRUST_CPP_DIALECT >= 2011

#include <unittest/unittest.h>

#include <thrust/limits.h>
#include <thrust/async/copy.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#define DEFINE_ASYNC_COPY_CALLABLE(name, ...)                                 \
  struct THRUST_PP_CAT2(name, _fn)                                            \
  {                                                                           \
    template <typename ForwardIt, typename Sentinel, typename OutputIt>       \
    __host__                                                                  \
    auto operator()(                                                          \
      ForwardIt&& first, Sentinel&& last, OutputIt&& output                   \
    ) const                                                                   \
    THRUST_DECLTYPE_RETURNS(                                                  \
      ::thrust::async::copy(                                                  \
        __VA_ARGS__                                                           \
        THRUST_PP_COMMA_IF(THRUST_PP_ARITY(__VA_ARGS__))                      \
        THRUST_FWD(first), THRUST_FWD(last), THRUST_FWD(output)               \
      )                                                                       \
    )                                                                         \
  };                                                                          \
  /**/

DEFINE_ASYNC_COPY_CALLABLE(
  invoke_async_copy
);

DEFINE_ASYNC_COPY_CALLABLE(
  invoke_async_copy_host,   thrust::host
);
DEFINE_ASYNC_COPY_CALLABLE(
  invoke_async_copy_device, thrust::device
);

DEFINE_ASYNC_COPY_CALLABLE(
  invoke_async_copy_host_to_device,    thrust::host,   thrust::device
);
DEFINE_ASYNC_COPY_CALLABLE(
  invoke_async_copy_device_to_host,    thrust::device, thrust::host
);
DEFINE_ASYNC_COPY_CALLABLE(
  invoke_async_copy_host_to_host,      thrust::host,   thrust::host
);
DEFINE_ASYNC_COPY_CALLABLE(
  invoke_async_copy_device_to_device,  thrust::device, thrust::device
);

#undef DEFINE_ASYNC_COPY_CALLABLE

///////////////////////////////////////////////////////////////////////////////

template <typename AsyncCopyCallable>
struct test_async_copy_host_to_device
{
  template <typename T>
  struct tester
  {
    __host__
    void operator()(std::size_t n)
    {
      thrust::host_vector<T>   h0_data(unittest::random_integers<T>(n));
      thrust::device_vector<T> d0_data(n);

      auto f0 = AsyncCopyCallable{}(
        h0_data.begin(), h0_data.end(), d0_data.begin()
      );

      f0.wait();

      ASSERT_EQUAL(h0_data, d0_data);
    }
  };
};
DECLARE_VARIABLE_UNITTEST_WITH_TYPES_AND_NAME(
  test_async_copy_host_to_device<invoke_async_copy_fn>::tester
, BuiltinNumericTypes
, test_async_copy_trivially_relocatable_elements_host_to_device
);
DECLARE_VARIABLE_UNITTEST_WITH_TYPES_AND_NAME(
  test_async_copy_host_to_device<invoke_async_copy_host_to_device_fn>::tester
, BuiltinNumericTypes
, test_async_copy_trivially_relocatable_elements_host_to_device_policies
);

///////////////////////////////////////////////////////////////////////////////

template <typename AsyncCopyCallable>
struct test_async_copy_device_to_host
{
  template <typename T>
  struct tester
  {
    __host__
    void operator()(std::size_t n)
    {
      thrust::host_vector<T>   h0_data(unittest::random_integers<T>(n));
      thrust::device_vector<T> h1_data(n);
      thrust::device_vector<T> d0_data(n);

      thrust::copy(h0_data.begin(), h0_data.end(), d0_data.begin());

      ASSERT_EQUAL(h0_data, d0_data);

      auto f0 = AsyncCopyCallable{}(
        d0_data.begin(), d0_data.end(), h1_data.begin()
      );

      f0.wait();

      ASSERT_EQUAL(h0_data, d0_data);
      ASSERT_EQUAL(d0_data, h1_data);
    }
  };
};
DECLARE_VARIABLE_UNITTEST_WITH_TYPES_AND_NAME(
  test_async_copy_device_to_host<invoke_async_copy_fn>::tester
, BuiltinNumericTypes
, test_async_copy_trivially_relocatable_elements_device_to_host
);
DECLARE_VARIABLE_UNITTEST_WITH_TYPES_AND_NAME(
  test_async_copy_device_to_host<invoke_async_copy_device_to_host_fn>::tester
, BuiltinNumericTypes
, test_async_copy_trivially_relocatable_elements_device_to_host_policies
);

///////////////////////////////////////////////////////////////////////////////

template <typename AsyncCopyCallable>
struct test_async_copy_device_to_device
{
  template <typename T>
  struct tester
  {
    __host__
    void operator()(std::size_t n)
    {
      thrust::host_vector<T>   h0_data(unittest::random_integers<T>(n));
      thrust::device_vector<T> d0_data(n);
      thrust::device_vector<T> d1_data(n);

      thrust::copy(h0_data.begin(), h0_data.end(), d0_data.begin());

      ASSERT_EQUAL(h0_data, d0_data);

      auto f0 = AsyncCopyCallable{}(
        d0_data.begin(), d0_data.end(), d1_data.begin()
      );

      f0.wait();

      ASSERT_EQUAL(h0_data, d0_data);
      ASSERT_EQUAL(d0_data, d1_data);
    }
  };
};
DECLARE_VARIABLE_UNITTEST_WITH_TYPES_AND_NAME(
  test_async_copy_device_to_device<invoke_async_copy_fn>::tester
, NumericTypes
, test_async_copy_device_to_device
);
DECLARE_VARIABLE_UNITTEST_WITH_TYPES_AND_NAME(
  test_async_copy_device_to_device<invoke_async_copy_device_fn>::tester
, NumericTypes
, test_async_copy_device_to_device_policy
);
DECLARE_VARIABLE_UNITTEST_WITH_TYPES_AND_NAME(
  test_async_copy_device_to_device<invoke_async_copy_device_to_device_fn>::tester
, NumericTypes
, test_async_copy_device_to_device_policies
);

///////////////////////////////////////////////////////////////////////////////

// Non ContiguousIterator input.
template <typename AsyncCopyCallable>
struct test_async_copy_counting_iterator_input_to_device_vector
{
  template <typename T>
  struct tester
  {
    __host__
    void operator()(std::size_t n)
    {
      thrust::counting_iterator<T> first(0);
      thrust::counting_iterator<T> last(
        unittest::truncate_to_max_representable<T>(n)
      );

      thrust::device_vector<T> d0_data(n);
      thrust::device_vector<T> d1_data(n);

      thrust::copy(first, last, d0_data.begin());

      auto f0 = AsyncCopyCallable{}(
        first, last, d1_data.begin()
      );

      f0.wait();

      ASSERT_EQUAL(d0_data, d1_data);
    }
  };
};
DECLARE_VARIABLE_UNITTEST_WITH_TYPES_AND_NAME(
  test_async_copy_counting_iterator_input_to_device_vector<
    invoke_async_copy_fn
  >::tester
, BuiltinNumericTypes
, test_async_copy_counting_iterator_input_trivially_relocatable_elements_device_to_device
);
DECLARE_VARIABLE_UNITTEST_WITH_TYPES_AND_NAME(
  test_async_copy_counting_iterator_input_to_device_vector<
    invoke_async_copy_device_fn
  >::tester
, BuiltinNumericTypes
, test_async_copy_counting_iterator_input_trivially_relocatable_elements_device_to_device_policy
);
DECLARE_VARIABLE_UNITTEST_WITH_TYPES_AND_NAME(
  test_async_copy_counting_iterator_input_to_device_vector<
    invoke_async_copy_device_to_device_fn
  >::tester
, BuiltinNumericTypes
, test_async_copy_counting_iterator_input_trivially_relocatable_elements_device_to_device_policies
);
DECLARE_VARIABLE_UNITTEST_WITH_TYPES_AND_NAME(
  test_async_copy_counting_iterator_input_to_device_vector<
    invoke_async_copy_host_to_device_fn
  >::tester
  // TODO: Re-add custom_numeric when it supports counting iterators.
, BuiltinNumericTypes
, test_async_copy_counting_iterator_input_host_to_device_policies
);

///////////////////////////////////////////////////////////////////////////////

// Non ContiguousIterator input.
template <typename AsyncCopyCallable>
struct test_async_copy_counting_iterator_input_to_host_vector
{
  template <typename T>
  struct tester
  {
    __host__
    void operator()(std::size_t n)
    {
      thrust::counting_iterator<T> first(0);
      thrust::counting_iterator<T> last(
        unittest::truncate_to_max_representable<T>(n)
      );

      thrust::host_vector<T> d0_data(n);
      thrust::host_vector<T> d1_data(n);

      thrust::copy(first, last, d0_data.begin());

      auto f0 = AsyncCopyCallable{}(
        first, last, d1_data.begin()
      );

      f0.wait();

      ASSERT_EQUAL(d0_data, d1_data);
    }
  };
};
DECLARE_VARIABLE_UNITTEST_WITH_TYPES_AND_NAME(
  test_async_copy_counting_iterator_input_to_host_vector<
    invoke_async_copy_fn
  >::tester
, BuiltinNumericTypes
, test_async_copy_counting_iterator_input_trivially_relocatable_elements_device_to_host
);
DECLARE_VARIABLE_UNITTEST_WITH_TYPES_AND_NAME(
  test_async_copy_counting_iterator_input_to_host_vector<
    invoke_async_copy_device_to_host_fn
  >::tester
, BuiltinNumericTypes
, test_async_copy_counting_iterator_input_trivially_relocatable_elements_device_to_host_policies
);

///////////////////////////////////////////////////////////////////////////////

// TODO: device_to_device NonContiguousIterator output (discard_iterator).

// TODO: host_to_device non trivially relocatable.

// TODO: device_to_host non trivially relocatable.

// TODO: host_to_device NonContiguousIterator input (counting_iterator).

// TODO: host_to_device NonContiguousIterator output (discard_iterator).

// TODO: device_to_host NonContiguousIterator input (counting_iterator).

// TODO: device_to_host NonContiguousIterator output (discard_iterator).

// TODO: Mixed types, needs loosening of `is_trivially_relocatable_to` logic.

// TODO: H->D copy, then dependent D->H copy (round trip).
// Can't do this today because we can't do cross-system with explicit policies.

#endif // THRUST_CPP_DIALECT >= 2011

