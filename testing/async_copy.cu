#include <thrust/detail/config.h>

#if THRUST_CPP_DIALECT >= 2011

#include <unittest/unittest.h>

#include <thrust/async/copy.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

template <typename T>
__host__
void
test_async_copy_host_to_device_trivially_relocatable(
  std::size_t n
)
{
  thrust::host_vector<T>   h0_data(unittest::random_integers<T>(n));
  thrust::device_vector<T> d0_data(n);

  auto f0 = thrust::async::copy(
    h0_data.begin(), h0_data.end(), d0_data.begin()
  );

  std::move(f0).get();

  ASSERT_EQUAL(h0_data, d0_data);
}
DECLARE_VARIABLE_UNITTEST(
  test_async_copy_host_to_device_trivially_relocatable
);

template <typename T>
__host__
void
test_async_copy_host_to_device_trivially_relocatable_with_policies(
  std::size_t n
)
{
  thrust::host_vector<T>   h0_data(unittest::random_integers<T>(n));
  thrust::device_vector<T> d0_data(n);

  auto f0 = thrust::async::copy(
    thrust::host, thrust::device
  , h0_data.begin(), h0_data.end(), d0_data.begin()
  );

  std::move(f0).get();

  ASSERT_EQUAL(h0_data, d0_data);
}
DECLARE_VARIABLE_UNITTEST(
  test_async_copy_host_to_device_trivially_relocatable_with_policies
);

template <typename T>
__host__
void
test_async_copy_device_to_host_trivially_relocatable(
  std::size_t n
)
{
  thrust::host_vector<T>   h0_data(unittest::random_integers<T>(n));
  thrust::device_vector<T> h1_data(n);
  thrust::device_vector<T> d0_data(n);

  thrust::copy(h0_data.begin(), h0_data.end(), d0_data.begin());

  ASSERT_EQUAL(h0_data, d0_data);

  auto f0 = thrust::async::copy(
    d0_data.begin(), d0_data.end(), h1_data.begin()
  );

  std::move(f0).get();

  ASSERT_EQUAL(h0_data, d0_data);
  ASSERT_EQUAL(d0_data, h1_data);
}
DECLARE_VARIABLE_UNITTEST(
  test_async_copy_device_to_host_trivially_relocatable
);

template <typename T>
__host__
void
test_async_copy_device_to_host_trivially_relocatable_with_policies(
  std::size_t n
)
{
  thrust::host_vector<T>   h0_data(unittest::random_integers<T>(n));
  thrust::device_vector<T> h1_data(n);
  thrust::device_vector<T> d0_data(n);

  thrust::copy(h0_data.begin(), h0_data.end(), d0_data.begin());

  ASSERT_EQUAL(h0_data, d0_data);

  auto f0 = thrust::async::copy(
    thrust::device, thrust::host
  , d0_data.begin(), d0_data.end(), h1_data.begin()
  );

  std::move(f0).get();

  ASSERT_EQUAL(h0_data, d0_data);
  ASSERT_EQUAL(d0_data, h1_data);
}
DECLARE_VARIABLE_UNITTEST(
  test_async_copy_device_to_host_trivially_relocatable_with_policies
);

template <typename T>
struct test_async_copy_device_to_device
{
  __host__
  void operator()(std::size_t n)
  {
    thrust::host_vector<T>   h0_data(unittest::random_integers<T>(n));
    thrust::device_vector<T> d0_data(n);
    thrust::device_vector<T> d1_data(n);

    thrust::copy(h0_data.begin(), h0_data.end(), d0_data.begin());

    ASSERT_EQUAL(h0_data, d0_data);

    auto f0 = thrust::async::copy(d0_data.begin(), d0_data.end(), d1_data.begin());

    std::move(f0).get();

    ASSERT_EQUAL(h0_data, d0_data);
    ASSERT_EQUAL(d0_data, d1_data);
  }
};
// TODO: Switch to `DECLARE_VARIABLE_UNITTEST` when we add `custom_numeric` to
// the list of types it covers.
VariableUnitTest<
  test_async_copy_device_to_device
, NumericTypes
> test_async_copy_device_to_device_instance;

template <typename T>
struct test_async_copy_device_to_device_with_policy
{
  __host__
  void operator()(std::size_t n)
  {
    thrust::host_vector<T>   h0_data(unittest::random_integers<T>(n));
    thrust::device_vector<T> d0_data(n);
    thrust::device_vector<T> d1_data(n);

    thrust::copy(h0_data.begin(), h0_data.end(), d0_data.begin());

    ASSERT_EQUAL(h0_data, d0_data);

    auto f0 = thrust::async::copy(
      thrust::device, d0_data.begin(), d0_data.end(), d1_data.begin()
    );

    std::move(f0).get();

    ASSERT_EQUAL(h0_data, d0_data);
    ASSERT_EQUAL(d0_data, d1_data);
  }
};
// TODO: Switch to `DECLARE_VARIABLE_UNITTEST` when we add `custom_numeric` to
// the list of types it covers.
VariableUnitTest<
  test_async_copy_device_to_device_with_policy
, NumericTypes
> test_async_copy_device_to_device_with_policy_instance;

// TODO: device_to_device implicit.

// TODO: device_to_device NonContiguousIterator input (counting_iterator).

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

