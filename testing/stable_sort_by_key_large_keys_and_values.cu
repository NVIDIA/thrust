#include <thrust/functional.h>
#include <thrust/sort.h>

#include <unittest/unittest.h>

template <unsigned int N>
void _TestStableSortByKeyWithLargeKeysAndValues()
{
  size_t n = (128 * 1024) / sizeof(FixedVector<int, N>);

  thrust::host_vector<FixedVector<int, N>> h_keys(n);
  thrust::host_vector<FixedVector<int, N>> h_vals(n);

  for (size_t i = 0; i < n; i++)
  {
    const auto uint_i   = static_cast<unsigned int>(i);
    const auto rand_int = unittest::generate_random_integer<int>()(uint_i);
    h_keys[i]           = FixedVector<int, N>(rand_int);
    h_vals[i]           = FixedVector<int, N>(static_cast<int>(i));
  }

  thrust::device_vector<FixedVector<int, N>> d_keys = h_keys;
  thrust::device_vector<FixedVector<int, N>> d_vals = h_vals;

  thrust::stable_sort_by_key(h_keys.begin(), h_keys.end(), h_vals.begin());
  thrust::stable_sort_by_key(d_keys.begin(), d_keys.end(), d_vals.begin());

  ASSERT_EQUAL_QUIET(h_keys, d_keys);
  ASSERT_EQUAL_QUIET(h_vals, d_vals);
}

void TestStableSortByKeyWithLargeKeysAndValues()
{
  _TestStableSortByKeyWithLargeKeysAndValues<4>();
  _TestStableSortByKeyWithLargeKeysAndValues<8>();
  _TestStableSortByKeyWithLargeKeysAndValues<16>();
}
DECLARE_UNITTEST(TestStableSortByKeyWithLargeKeysAndValues);
