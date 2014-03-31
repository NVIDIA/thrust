#include <unittest/unittest.h>
#include <thrust/generate.h>
#include <thrust/execution_policy.h>


template<typename Iterator, typename Function>
__global__
void generate_kernel(Iterator first, Iterator last, Function f)
{
  thrust::generate(thrust::seq, first, last, f);
}


template<typename T>
struct return_value
{
  T val;
  
  return_value(void){}
  return_value(T v):val(v){}
  
  __host__ __device__
  T operator()(void){ return val; }
};


template<typename T>
void TestGenerateDeviceSeq(const size_t n)
{
  thrust::host_vector<T> h_result(n);
  thrust::device_vector<T> d_result(n);
  
  T value = 13;
  return_value<T> f(value);
  
  thrust::generate(h_result.begin(), h_result.end(), f);
  generate_kernel<<<1,1>>>(d_result.begin(), d_result.end(), f);
  
  ASSERT_EQUAL(h_result, d_result);
}
DECLARE_VARIABLE_UNITTEST(TestGenerateDeviceSeq);


void TestGenerateCudaStreams()
{
  thrust::device_vector<int> result(5);
  
  int value = 13;
  
  return_value<int> f(value);

  cudaStream_t s;
  cudaStreamCreate(&s);
  
  thrust::generate(thrust::cuda::par(s), result.begin(), result.end(), f);
  cudaStreamSynchronize(s);
  
  ASSERT_EQUAL(result[0], value);
  ASSERT_EQUAL(result[1], value);
  ASSERT_EQUAL(result[2], value);
  ASSERT_EQUAL(result[3], value);
  ASSERT_EQUAL(result[4], value);

  cudaStreamDestroy(s);
}
DECLARE_UNITTEST(TestGenerateCudaStreams);


template<typename Iterator, typename Size, typename Function>
__global__
void generate_n_kernel(Iterator first, Size n, Function f)
{
  thrust::generate_n(thrust::seq, first, n, f);
}


template<typename T>
void TestGenerateNDeviceSeq(const size_t n)
{
  thrust::host_vector<T> h_result(n);
  thrust::device_vector<T> d_result(n);
  
  T value = 13;
  return_value<T> f(value);
  
  thrust::generate_n(h_result.begin(), h_result.size(), f);
  generate_n_kernel<<<1,1>>>(d_result.begin(), d_result.size(), f);
  
  ASSERT_EQUAL(h_result, d_result);
}
DECLARE_VARIABLE_UNITTEST(TestGenerateNDeviceSeq);


void TestGenerateNCudaStreams()
{
  thrust::device_vector<int> result(5);
  
  int value = 13;
  
  return_value<int> f(value);

  cudaStream_t s;
  cudaStreamCreate(&s);
  
  thrust::generate_n(thrust::cuda::par(s), result.begin(), result.size(), f);
  cudaStreamSynchronize(s);
  
  ASSERT_EQUAL(result[0], value);
  ASSERT_EQUAL(result[1], value);
  ASSERT_EQUAL(result[2], value);
  ASSERT_EQUAL(result[3], value);
  ASSERT_EQUAL(result[4], value);

  cudaStreamDestroy(s);
}
DECLARE_UNITTEST(TestGenerateNCudaStreams);

