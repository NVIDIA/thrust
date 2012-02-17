#include <unittest/unittest.h>
#include <thrust/merge.h>
#include <thrust/functional.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/unique.h>
#include <thrust/iterator/discard_iterator.h>

template<typename Vector>
void TestMergeByKeySimple(void)
{
  typedef typename Vector::iterator Iterator;

  Vector a(3), b(4);
  Vector u(3), v(4);

  a[0] = 0; a[1] = 2; a[2] = 4;
  b[0] = 0; b[1] = 3; b[2] = 3; b[3] = 4;
  
  u[0] = 1; u[1] = 2; u[2] = 3;
  v[0] = 4; v[1] = 5; v[2] = 6; v[3] = 7;

  Vector ref_keys(7);
  ref_keys[0] = 0;
  ref_keys[1] = 0;
  ref_keys[2] = 2;
  ref_keys[3] = 3;
  ref_keys[4] = 3;
  ref_keys[5] = 4;
  ref_keys[6] = 4;
  
  Vector ref_vals(7);
  ref_vals[0] = 1;
  ref_vals[1] = 4;
  ref_vals[2] = 2;
  ref_vals[3] = 5;
  ref_vals[4] = 6;
  ref_vals[5] = 3;
  ref_vals[6] = 7;

  Vector keys_out(7);
  Vector vals_out(7);

  thrust::pair<Iterator,Iterator> result = thrust::system::tbb::detail::merge_by_key
    (thrust::system::tbb::tag(),
     a.begin(), a.end(),
     b.begin(), b.end(),
     u.begin(),
     v.begin(),
     keys_out.begin(),
     vals_out.begin(),
     thrust::less<typename Vector::value_type>());

  ASSERT_EQUAL_QUIET(result.first,  keys_out.end());
  ASSERT_EQUAL_QUIET(result.second, vals_out.end());
  ASSERT_EQUAL(keys_out, ref_keys);
  ASSERT_EQUAL(vals_out, ref_vals);
}
DECLARE_VECTOR_UNITTEST(TestMergeByKeySimple);

template<typename T>
  void TestMergeByKey(size_t n)
{
  size_t sizes[]   = {0, 1, n / 2, n, n + 1, 2 * n};
  size_t num_sizes = sizeof(sizes) / sizeof(size_t);

  thrust::host_vector<T> random = unittest::random_integers<char>(n + *thrust::max_element(sizes, sizes + num_sizes));

  thrust::host_vector<T> h_a(random.begin(), random.begin() + n);
  thrust::host_vector<T> h_b(random.begin() + n, random.end());

  thrust::stable_sort(h_a.begin(), h_a.end());
  thrust::stable_sort(h_b.begin(), h_b.end());
  
  thrust::host_vector<int> h_u(h_a.size()); thrust::sequence(h_u.begin(), h_u.end());
  thrust::host_vector<int> h_v(h_b.size()); thrust::sequence(h_v.begin(), h_v.end(), (int) h_u.size());
  
  thrust::device_vector<T> d_a = h_a;
  thrust::device_vector<T> d_b = h_b;
  thrust::device_vector<int> d_u(h_u);
  thrust::device_vector<int> d_v(h_v);

  for (size_t i = 0; i < num_sizes; i++)
  {
    size_t size = sizes[i];
    
    thrust::host_vector<T>   h_key_result(n + size);
    thrust::device_vector<T> d_key_result(n + size);
    
    thrust::host_vector<int>   h_val_result(n + size);
    thrust::device_vector<int> d_val_result(n + size);

    typename thrust::host_vector<T>::iterator   h_end;
    typename thrust::device_vector<T>::iterator d_end;
  
    h_end = thrust::system::tbb::detail::merge_by_key
    (thrust::system::tbb::tag(),
     h_a.begin(), h_a.end(),
     h_b.begin(), h_b.begin() + size,
     h_u.begin(),
     h_v.begin(),
     h_key_result.begin(),
     h_val_result.begin(),
     thrust::less<T>()).first;

    h_key_result.resize(h_end - h_key_result.begin());
    h_val_result.resize(h_end - h_key_result.begin());

    d_end = thrust::system::tbb::detail::merge_by_key
    (thrust::system::tbb::tag(),
     d_a.begin(), d_a.end(),
     d_b.begin(), d_b.begin() + size,
     d_u.begin(),
     d_v.begin(),
     d_key_result.begin(),
     d_val_result.begin(),
     thrust::less<T>()).first;
    d_key_result.resize(d_end - d_key_result.begin());
    d_val_result.resize(d_end - d_key_result.begin());

    ASSERT_EQUAL(h_key_result, d_key_result);
    ASSERT_EQUAL(h_val_result, d_val_result);
  }
}
DECLARE_VARIABLE_UNITTEST(TestMergeByKey);

template<typename T>
void TestMergeByKeyDescending(size_t n)
{
  thrust::host_vector<T> h_a = unittest::random_integers<T>(n);
  thrust::host_vector<T> h_b = unittest::random_integers<T>(n);
  thrust::host_vector<int> h_u(h_a.size()); thrust::sequence(h_u.begin(), h_u.end());
  thrust::host_vector<int> h_v(h_b.size()); thrust::sequence(h_v.begin(), h_v.end(), (int) h_u.size());

  thrust::stable_sort(h_a.begin(), h_a.end(), thrust::greater<T>());
  thrust::stable_sort(h_b.begin(), h_b.end(), thrust::greater<T>());

  thrust::device_vector<T> d_a = h_a;
  thrust::device_vector<T> d_b = h_b;
  thrust::device_vector<int> d_u(h_u);
  thrust::device_vector<int> d_v(h_v);

  thrust::host_vector<T>     h_key_result(h_a.size() + h_b.size());
  thrust::device_vector<T>   d_key_result(d_a.size() + d_b.size());
  thrust::host_vector<int>   h_val_result(h_a.size() + h_b.size());
  thrust::device_vector<int> d_val_result(d_a.size() + d_b.size());

  thrust::system::tbb::detail::merge_by_key
    (thrust::system::tbb::tag(),
     h_a.begin(), h_a.end(),
     h_b.begin(), h_b.end(),
     h_u.begin(),
     h_v.begin(),
     h_key_result.begin(),
     h_val_result.begin(),
     thrust::greater<T>());

  thrust::system::tbb::detail::merge_by_key
    (thrust::system::tbb::tag(),
     d_a.begin(), d_a.end(),
     d_b.begin(), d_b.end(),
     d_u.begin(),
     d_v.begin(),
     d_key_result.begin(),
     d_val_result.begin(),
     thrust::greater<T>());

  ASSERT_EQUAL(h_key_result, d_key_result);
  ASSERT_EQUAL(h_val_result, d_val_result);
}
DECLARE_VARIABLE_UNITTEST(TestMergeByKeyDescending);

// TODO add TestMergeByKeyToDiscardIterator

