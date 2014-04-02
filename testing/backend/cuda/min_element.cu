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


void TestMinElementCudaStreams()
{
  typedef thrust::device_vector<int> Vector;
  typedef typename Vector::value_type T;

  Vector data(6);
  data[0] = 3;
  data[1] = 5;
  data[2] = 1;
  data[3] = 2;
  data[4] = 5;
  data[5] = 1;

  cudaStream_t s;
  cudaStreamCreate(&s);

  ASSERT_EQUAL( *thrust::min_element(thrust::cuda::par(s), data.begin(), data.end()), 1);
  ASSERT_EQUAL( thrust::min_element(thrust::cuda::par(s), data.begin(), data.end()) - data.begin(), 2);
  
  ASSERT_EQUAL( *thrust::min_element(thrust::cuda::par(s), data.begin(), data.end(), thrust::greater<T>()), 5);
  ASSERT_EQUAL( thrust::min_element(thrust::cuda::par(s), data.begin(), data.end(), thrust::greater<T>()) - data.begin(), 1);

  cudaStreamDestroy(s);
}
DECLARE_UNITTEST(TestMinElementCudaStreams);

