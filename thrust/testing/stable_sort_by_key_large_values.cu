#include <thrust/functional.h>
#include <thrust/sort.h>

#include <unittest/unittest.h>

template <typename T>
struct greater_div_10
{
  __host__ __device__ bool operator()(const T &lhs, const T &rhs) const
  {
    return ((int)lhs) / 10 > ((int)rhs) / 10;
  }
};

template <unsigned int N>
void _TestStableSortByKeyWithLargeValues()
{
  size_t n = (128 * 1024) / sizeof(FixedVector<int, N>);

  thrust::host_vector<unsigned int> h_keys(n);
  thrust::host_vector<FixedVector<int, N>> h_vals(n);

  for (size_t i = 0; i < n; i++)
  {
    const auto uint_i   = static_cast<unsigned int>(i);
    const auto rand_int = unittest::generate_random_integer<unsigned int>()(uint_i);
    h_keys[i]           = rand_int;
    h_vals[i]           = FixedVector<int, N>(static_cast<int>(i));
  }

  thrust::device_vector<unsigned int> d_keys        = h_keys;
  thrust::device_vector<FixedVector<int, N>> d_vals = h_vals;

  thrust::stable_sort_by_key(h_keys.begin(), h_keys.end(), h_vals.begin());
  thrust::stable_sort_by_key(d_keys.begin(), d_keys.end(), d_vals.begin());

  ASSERT_EQUAL_QUIET(h_keys, d_keys);
  ASSERT_EQUAL_QUIET(h_vals, d_vals);

  // so cuda::stable_merge_sort_by_key() is called
  thrust::stable_sort_by_key(h_keys.begin(),
                             h_keys.end(),
                             h_vals.begin(),
                             greater_div_10<unsigned int>());
  thrust::stable_sort_by_key(d_keys.begin(),
                             d_keys.end(),
                             d_vals.begin(),
                             greater_div_10<unsigned int>());

  ASSERT_EQUAL_QUIET(h_keys, d_keys);
  ASSERT_EQUAL_QUIET(h_vals, d_vals);
}

void TestStableSortByKeyWithLargeValues()
{
  _TestStableSortByKeyWithLargeValues<4>();
  _TestStableSortByKeyWithLargeValues<8>();
  _TestStableSortByKeyWithLargeValues<16>();
}
DECLARE_UNITTEST(TestStableSortByKeyWithLargeValues);
