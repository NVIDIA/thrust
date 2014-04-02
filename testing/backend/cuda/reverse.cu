#include <unittest/unittest.h>
#include <thrust/reverse.h>
#include <thrust/execution_policy.h>


template<typename Iterator>
__global__
void reverse_kernel(Iterator first, Iterator last)
{
  thrust::reverse(thrust::seq, first, last);
}


template<typename T>
void TestReverseDeviceSeq(const size_t n)
{
  thrust::host_vector<T> h_data = unittest::random_integers<T>(n);
  thrust::device_vector<T> d_data = h_data;
  
  thrust::reverse(h_data.begin(), h_data.end());
  reverse_kernel<<<1,1>>>(d_data.begin(), d_data.end());
  
  ASSERT_EQUAL(h_data, d_data);
};
DECLARE_VARIABLE_UNITTEST(TestReverseDeviceSeq);


template<typename Iterator1, typename Iterator2>
__global__
void reverse_copy_kernel(Iterator1 first, Iterator1 last, Iterator2 result)
{
  thrust::reverse_copy(thrust::seq, first, last, result);
}


template<typename T>
void TestReverseCopyDeviceSeq(const size_t n)
{
  thrust::host_vector<T> h_data = unittest::random_integers<T>(n);
  thrust::device_vector<T> d_data = h_data;

  thrust::host_vector<T> h_result(n);
  thrust::device_vector<T> d_result(n);

  thrust::reverse_copy(h_data.begin(), h_data.end(), h_result.begin());
  reverse_copy_kernel<<<1,1>>>(d_data.begin(), d_data.end(), d_result.begin());

  ASSERT_EQUAL(h_result, d_result);
};
DECLARE_VARIABLE_UNITTEST(TestReverseCopyDeviceSeq);


void TestReverseCudaStreams()
{
  typedef thrust::device_vector<int> Vector;
  Vector data(5);
  data[0] = 1;
  data[1] = 2;
  data[2] = 3;
  data[3] = 4;
  data[4] = 5;

  cudaStream_t s;
  cudaStreamCreate(&s);

  thrust::reverse(thrust::cuda::par(s), data.begin(), data.end());

  cudaStreamSynchronize(s);

  Vector ref(5);
  ref[0] = 5;
  ref[1] = 4;
  ref[2] = 3;
  ref[3] = 2;
  ref[4] = 1;

  ASSERT_EQUAL(ref, data);

  cudaStreamDestroy(s);
}
DECLARE_UNITTEST(TestReverseCudaStreams);

