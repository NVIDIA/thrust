#include <unittest/unittest.h>
#include <thrust/merge.h>
#include <thrust/functional.h>
#include <thrust/sort.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>


template<typename Iterator1, typename Iterator2, typename Iterator3, typename Iterator4>
__global__
void merge_kernel(Iterator1 first1, Iterator1 last1,
                  Iterator2 first2, Iterator2 last2,
                  Iterator3 result1,
                  Iterator4 result2)
{
  *result2 = thrust::merge(thrust::seq, first1, last1, first2, last2, result1);
}


template<typename T>
  void TestMergeDeviceSeq(size_t n)
{
  size_t sizes[]   = {0, 1, n / 2, n, n + 1, 2 * n};
  size_t num_sizes = sizeof(sizes) / sizeof(size_t);

  thrust::host_vector<T> random = unittest::random_integers<unittest::int8_t>(n + *thrust::max_element(sizes, sizes + num_sizes));

  thrust::host_vector<T> h_a(random.begin(), random.begin() + n);
  thrust::host_vector<T> h_b(random.begin() + n, random.end());

  thrust::stable_sort(h_a.begin(), h_a.end());
  thrust::stable_sort(h_b.begin(), h_b.end());
  
  thrust::device_vector<T> d_a = h_a;
  thrust::device_vector<T> d_b = h_b;

  for(size_t i = 0; i < num_sizes; i++)
  {
    size_t size = sizes[i];
    
    thrust::host_vector<T>   h_result(n + size);
    thrust::device_vector<T> d_result(n + size);

    typename thrust::host_vector<T>::iterator   h_end;

    typedef typename thrust::device_vector<T>::iterator iter_type;
    thrust::device_vector<iter_type> d_end(1);
    
    h_end = thrust::merge(h_a.begin(), h_a.end(),
                          h_b.begin(), h_b.begin() + size,
                          h_result.begin());
    h_result.resize(h_end - h_result.begin());

    merge_kernel<<<1,1>>>(d_a.begin(), d_a.end(),
                          d_b.begin(), d_b.begin() + size,
                          d_result.begin(),
                          d_end.begin());
    d_result.resize((iter_type)d_end[0] - d_result.begin());

    ASSERT_EQUAL(h_result, d_result);
  }
}
DECLARE_VARIABLE_UNITTEST(TestMergeDeviceSeq);

