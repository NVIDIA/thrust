#include <thrust/detail/config.h>

#if THRUST_CPP_DIALECT >= 2011

#include <unittest/unittest.h>

#include <thrust/async/sort.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

template <typename T>
struct custom_greater
{
  __host__ __device__
  bool operator()(T rhs, T lhs) const
  {
    return lhs > rhs;
  }
};

template <typename T>
struct test_async_sort
{
  __host__
  void operator()(std::size_t n)
  {
    thrust::host_vector<T>   h0_data(unittest::random_integers<T>(n));
    thrust::device_vector<T> d0_data(h0_data);

    ASSERT_EQUAL(h0_data, d0_data);

    thrust::sort(
      h0_data.begin(), h0_data.end()
    );

    auto f0 = thrust::async::sort(
      d0_data.begin(), d0_data.end()
    );

    f0.wait();

    ASSERT_EQUAL(h0_data, d0_data);
  }
};
// TODO: Switch to `DECLARE_VARIABLE_UNITTEST` when we add `custom_numeric` to
// the list of types it covers.
VariableUnitTest<
  test_async_sort
, NumericTypes
> test_async_sort_instance;

template <typename T>
struct test_async_sort_policy
{
  __host__
  void operator()(std::size_t n)
  {
    thrust::host_vector<T>   h0_data(unittest::random_integers<T>(n));
    thrust::device_vector<T> d0_data(h0_data);

    ASSERT_EQUAL(h0_data, d0_data);

    thrust::sort(
      h0_data.begin(), h0_data.end()
    );

    auto f0 = thrust::async::sort(
      thrust::device, d0_data.begin(), d0_data.end()
    );

    f0.wait();

    ASSERT_EQUAL(h0_data, d0_data);
  }
};
// TODO: Switch to `DECLARE_VARIABLE_UNITTEST` when we add `custom_numeric` to
// the list of types it covers.
VariableUnitTest<
  test_async_sort_policy
, NumericTypes
> test_async_sort_policy_instance;

template <template <typename> class Op>
struct test_async_sort_op
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

      Op<T> op{};

      thrust::sort(
        h0_data.begin(), h0_data.end(), op
      );

      auto f0 = thrust::async::sort(
        d0_data.begin(), d0_data.end(), op
      );

      f0.wait();

      ASSERT_EQUAL(h0_data, d0_data);
    }
  };
};
// TODO: Switch to `DECLARE_VARIABLE_UNITTEST` when we add `custom_numeric` to
// the list of types it covers.
VariableUnitTest<
  test_async_sort_op<custom_greater>::tester
, NumericTypes
> test_async_sort_op_instance(
  "test_async_sort_op<custom_greater>"
);
VariableUnitTest<
  test_async_sort_op<thrust::less>::tester
, NumericTypes
> test_async_sort_less_instance(
  "test_async_sort_op<thrust::less>"
);
VariableUnitTest<
  test_async_sort_op<thrust::greater>::tester
, NumericTypes
> test_async_sort_greater_instance(
  "test_async_sort_op<thrust::greater>"
);

template <template <typename> class Op>
struct test_async_sort_policy_op
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

      Op<T> op{};

      thrust::sort(
        h0_data.begin(), h0_data.end(), op
      );

      auto f0 = thrust::async::sort(
        thrust::device, d0_data.begin(), d0_data.end(), op
      );

      f0.wait();

      ASSERT_EQUAL(h0_data, d0_data);
    }
  };
};
// TODO: Switch to `DECLARE_VARIABLE_UNITTEST` when we add `custom_numeric` to
// the list of types it covers.
VariableUnitTest<
  test_async_sort_policy_op<custom_greater>::tester
, NumericTypes
> test_async_sort_policy_op_instance(
  "test_async_sort_policy_op<custom_greater>"
);
VariableUnitTest<
  test_async_sort_policy_op<thrust::less>::tester
, NumericTypes
> test_async_sort_policy_less_instance(
  "test_async_sort_policy_op<thrust::less>"
);
VariableUnitTest<
  test_async_sort_policy_op<thrust::greater>::tester
, NumericTypes
> test_async_sort_policy_greater_instance(
  "test_async_sort_policy_op<thrust::greater>"
);

// TODO: Async copy then sort.

// TODO: Test future return type.

#endif // THRUST_CPP_DIALECT >= 2011

