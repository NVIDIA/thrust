#include <unittest/unittest.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>


template<typename ExecutionPolicy, typename Iterator, typename Iterator2>
__global__
void max_element_kernel(ExecutionPolicy exec, Iterator first, Iterator last, Iterator2 result)
{
  *result = thrust::max_element(exec, first, last);
}


template<typename ExecutionPolicy, typename Iterator, typename BinaryPredicate, typename Iterator2>
__global__
void max_element_kernel(ExecutionPolicy exec, Iterator first, Iterator last, BinaryPredicate pred, Iterator2 result)
{
  *result = thrust::max_element(exec, first, last, pred);
}


template<typename T, typename ExecutionPolicy>
void TestMaxElementDevice(ExecutionPolicy exec, const size_t n)
{
  thrust::host_vector<T> h_data = unittest::random_samples<T>(n);
  thrust::device_vector<T> d_data = h_data;

  typedef typename thrust::device_vector<T>::iterator iter_type;

  thrust::device_vector<iter_type> d_result(1);
  
  typename thrust::host_vector<T>::iterator   h_max = thrust::max_element(h_data.begin(), h_data.end());

  max_element_kernel<<<1,1>>>(exec, d_data.begin(), d_data.end(), d_result.begin());
  ASSERT_EQUAL(h_max - h_data.begin(), (iter_type)d_result[0] - d_data.begin());

  
  typename thrust::host_vector<T>::iterator   h_min = thrust::max_element(h_data.begin(), h_data.end(), thrust::greater<T>());

  max_element_kernel<<<1,1>>>(exec, d_data.begin(), d_data.end(), thrust::greater<T>(), d_result.begin());
  ASSERT_EQUAL(h_min - h_data.begin(), (iter_type)d_result[0] - d_data.begin());
}


template<typename T>
void TestMaxElementDeviceSeq(const size_t n)
{
  TestMaxElementDevice<T>(thrust::seq, n);
}
DECLARE_VARIABLE_UNITTEST(TestMaxElementDeviceSeq);


template<typename T>
void TestMaxElementDeviceDevice(const size_t n)
{
  TestMaxElementDevice<T>(thrust::device, n);
}
DECLARE_VARIABLE_UNITTEST(TestMaxElementDeviceDevice);


