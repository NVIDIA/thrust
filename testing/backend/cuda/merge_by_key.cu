#include <unittest/unittest.h>
#include <thrust/merge.h>
#include <thrust/functional.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>


template<typename Iterator1,
         typename Iterator2,
         typename Iterator3,
         typename Iterator4,
         typename Iterator5,
         typename Iterator6,
         typename Iterator7>
__global__
void merge_by_key_kernel(Iterator1 keys_first1, Iterator1 keys_last1,
                         Iterator2 keys_first2, Iterator2 keys_last2,
                         Iterator3 values_first1,
                         Iterator4 values_first2,
                         Iterator5 keys_result,
                         Iterator6 values_result,
                         Iterator7 result)
{
  *result = thrust::merge_by_key(thrust::seq, keys_first1, keys_last1, keys_first2, keys_last2, values_first1, values_first2, keys_result, values_result);
}


template<typename T>
void TestMergeByKeyDeviceSeq(size_t n)
{
  thrust::host_vector<T> random_keys = unittest::random_integers<unittest::int8_t>(n);
  thrust::host_vector<T> random_vals = unittest::random_integers<unittest::int8_t>(n);

  size_t denominators[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  size_t num_denominators = sizeof(denominators) / sizeof(size_t);

  for(size_t i = 0; i < num_denominators; ++i)
  {
    size_t size_a = n / denominators[i];

    thrust::host_vector<T> h_a_keys(random_keys.begin(), random_keys.begin() + size_a);
    thrust::host_vector<T> h_b_keys(random_keys.begin() + size_a, random_keys.end());

    thrust::host_vector<T> h_a_vals(random_vals.begin(), random_vals.begin() + size_a);
    thrust::host_vector<T> h_b_vals(random_vals.begin() + size_a, random_vals.end());

    thrust::stable_sort(h_a_keys.begin(), h_a_keys.end());
    thrust::stable_sort(h_b_keys.begin(), h_b_keys.end());

    thrust::device_vector<T> d_a_keys = h_a_keys;
    thrust::device_vector<T> d_b_keys = h_b_keys;

    thrust::device_vector<T> d_a_vals = h_a_vals;
    thrust::device_vector<T> d_b_vals = h_b_vals;

    thrust::host_vector<T> h_result_keys(n);
    thrust::host_vector<T> h_result_vals(n);

    thrust::device_vector<T> d_result_keys(n);
    thrust::device_vector<T> d_result_vals(n);

    thrust::pair<
      typename thrust::host_vector<T>::iterator,
      typename thrust::host_vector<T>::iterator
    > h_end;

    typedef thrust::pair<
      typename thrust::device_vector<T>::iterator,
      typename thrust::device_vector<T>::iterator
    > iter_pair_type;

    thrust::device_vector<iter_pair_type> d_end_vec(1);

    h_end = thrust::merge_by_key(h_a_keys.begin(), h_a_keys.end(),
                                 h_b_keys.begin(), h_b_keys.end(),
                                 h_a_vals.begin(),
                                 h_b_vals.begin(),
                                 h_result_keys.begin(),
                                 h_result_vals.begin());
    h_result_keys.erase(h_end.first, h_result_keys.end());
    h_result_vals.erase(h_end.second, h_result_vals.end());

    merge_by_key_kernel<<<1,1>>>(d_a_keys.begin(), d_a_keys.end(),
                                 d_b_keys.begin(), d_b_keys.end(),
                                 d_a_vals.begin(),
                                 d_b_vals.begin(),
                                 d_result_keys.begin(),
                                 d_result_vals.begin(),
                                 d_end_vec.begin());

    iter_pair_type d_end = d_end_vec[0];

    d_result_keys.erase(d_end.first, d_result_keys.end());
    d_result_vals.erase(d_end.second, d_result_vals.end());

    ASSERT_EQUAL(h_result_keys, d_result_keys);
    ASSERT_EQUAL(h_result_vals, d_result_vals);
  }
}
DECLARE_VARIABLE_UNITTEST(TestMergeByKeyDeviceSeq);

