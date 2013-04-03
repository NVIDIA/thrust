#include <unittest/unittest.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>


template<typename Iterator, typename Iterator2>
__global__
void min_element_kernel(Iterator first, Iterator last, Iterator2 result)
{
  *result = thrust::min_element(thrust::seq, first, last);
}


template<typename Iterator, typename BinaryPredicate, typename Iterator2>
__global__
void min_element_kernel(Iterator first, Iterator last, BinaryPredicate pred, Iterator2 result)
{
  *result = thrust::min_element(thrust::seq, first, last, pred);
}


template<typename T>
void TestMinElementDeviceSeq(const size_t n)
{
  thrust::host_vector<T> h_data = unittest::random_samples<T>(n);
  thrust::device_vector<T> d_data = h_data;

  typedef typename thrust::device_vector<T>::iterator iter_type;

  thrust::device_vector<iter_type> d_result(1);
  
  typename thrust::host_vector<T>::iterator   h_min = thrust::min_element(h_data.begin(), h_data.end());

  min_element_kernel<<<1,1>>>(d_data.begin(), d_data.end(), d_result.begin());
  ASSERT_EQUAL(h_min - h_data.begin(), (iter_type)d_result[0] - d_data.begin());

  
  typename thrust::host_vector<T>::iterator   h_max = thrust::min_element(h_data.begin(), h_data.end(), thrust::greater<T>());

  min_element_kernel<<<1,1>>>(d_data.begin(), d_data.end(), thrust::greater<T>(), d_result.begin());
  ASSERT_EQUAL(h_max - h_data.begin(), (iter_type)d_result[0] - d_data.begin());
}
DECLARE_VARIABLE_UNITTEST(TestMinElementDeviceSeq);

