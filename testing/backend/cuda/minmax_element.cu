#include <unittest/unittest.h>
#include <thrust/extrema.h>


template<typename Iterator1, typename Iterator2>
__global__
void minmax_element_kernel(Iterator1 first, Iterator1 last, Iterator2 result)
{
  *result = thrust::minmax_element(thrust::seq, first, last);
}


template<typename Iterator1, typename Iterator2, typename BinaryPredicate>
__global__
void minmax_element_kernel(Iterator1 first, Iterator1 last, BinaryPredicate pred, Iterator2 result)
{
  *result = thrust::minmax_element(thrust::seq, first, last, pred);
}


template<typename T>
void TestMinMaxElementDeviceSeq(const size_t n)
{
  thrust::host_vector<T>   h_data = unittest::random_samples<T>(n);
  thrust::device_vector<T> d_data = h_data;
  
  typename thrust::host_vector<T>::iterator   h_min;
  typename thrust::host_vector<T>::iterator   h_max;
  typename thrust::device_vector<T>::iterator d_min;
  typename thrust::device_vector<T>::iterator d_max;

  typedef thrust::pair<
    typename thrust::device_vector<T>::iterator,
    typename thrust::device_vector<T>::iterator
  > pair_type;

  thrust::device_vector<pair_type> d_result(1);
  
  h_min = thrust::minmax_element(h_data.begin(), h_data.end()).first;
  h_max = thrust::minmax_element(h_data.begin(), h_data.end()).second;

  d_min = thrust::minmax_element(d_data.begin(), d_data.end()).first;
  d_max = thrust::minmax_element(d_data.begin(), d_data.end()).second;

  minmax_element_kernel<<<1,1>>>(d_data.begin(), d_data.end(), d_result.begin());
  d_min = ((pair_type)d_result[0]).first;
  d_max = ((pair_type)d_result[0]).second;
  
  ASSERT_EQUAL(h_min - h_data.begin(), d_min - d_data.begin());
  ASSERT_EQUAL(h_max - h_data.begin(), d_max - d_data.begin());
  
  h_max = thrust::minmax_element(h_data.begin(), h_data.end(), thrust::greater<T>()).first;
  h_min = thrust::minmax_element(h_data.begin(), h_data.end(), thrust::greater<T>()).second;

  minmax_element_kernel<<<1,1>>>(d_data.begin(), d_data.end(), thrust::greater<T>(), d_result.begin());
  d_max = ((pair_type)d_result[0]).first;
  d_min = ((pair_type)d_result[0]).second;
  
  ASSERT_EQUAL(h_min - h_data.begin(), d_min - d_data.begin());
  ASSERT_EQUAL(h_max - h_data.begin(), d_max - d_data.begin());
}
DECLARE_VARIABLE_UNITTEST(TestMinMaxElementDeviceSeq);

