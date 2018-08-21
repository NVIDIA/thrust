#include <unittest/unittest.h>
#include <thrust/adjacent_difference.h>
#include <thrust/execution_policy.h>


template<typename ExecutionPolicy, typename Iterator1, typename Iterator2>
__global__ void adjacent_difference_kernel(ExecutionPolicy exec, Iterator1 first, Iterator1 last, Iterator2 result)
{
  thrust::adjacent_difference(exec, first, last, result);
}


template<typename ExecutionPolicy, typename Iterator1, typename Iterator2, typename BinaryFunction>
__global__ void adjacent_difference_kernel(ExecutionPolicy exec, Iterator1 first, Iterator1 last, Iterator2 result, BinaryFunction f)
{
  thrust::adjacent_difference(exec, first, last, result, f);
}


template<typename T, typename ExecutionPolicy>
void TestAdjacentDifferenceDevice(ExecutionPolicy exec, const size_t n)
{
  thrust::host_vector<T>   h_input = unittest::random_samples<T>(n);
  thrust::device_vector<T> d_input = h_input;
  
  thrust::host_vector<T>   h_output(n);
  thrust::device_vector<T> d_output(n);
  
  thrust::adjacent_difference(h_input.begin(), h_input.end(), h_output.begin());
  adjacent_difference_kernel<<<1,1>>>(exec, d_input.begin(), d_input.end(), d_output.begin());
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }
  
  ASSERT_EQUAL(h_output, d_output);
  
  thrust::adjacent_difference(h_input.begin(), h_input.end(), h_output.begin(), thrust::plus<T>());
  adjacent_difference_kernel<<<1,1>>>(exec, d_input.begin(), d_input.end(), d_output.begin(), thrust::plus<T>());
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }
  
  ASSERT_EQUAL(h_output, d_output);
  
  // in-place operation
  thrust::adjacent_difference(h_input.begin(), h_input.end(), h_input.begin(), thrust::plus<T>());
  adjacent_difference_kernel<<<1,1>>>(exec, d_input.begin(), d_input.end(), d_input.begin(), thrust::plus<T>());
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }
  
  ASSERT_EQUAL(h_input, h_output); //computed previously
  ASSERT_EQUAL(d_input, d_output); //computed previously
}


template<typename T>
void TestAdjacentDifferenceDeviceSeq(const size_t n)
{
  TestAdjacentDifferenceDevice<T>(thrust::seq, n);
}
DECLARE_VARIABLE_UNITTEST(TestAdjacentDifferenceDeviceSeq);


template<typename T>
void TestAdjacentDifferenceDeviceDevice(const size_t n)
{
  TestAdjacentDifferenceDevice<T>(thrust::device, n);
}
DECLARE_VARIABLE_UNITTEST(TestAdjacentDifferenceDeviceDevice);


void TestAdjacentDifferenceCudaStreams()
{
  cudaStream_t s;
  cudaStreamCreate(&s);
  
  thrust::device_vector<int> input(3);
  thrust::device_vector<int> output(3);
  input[0] = 1; input[1] = 4; input[2] = 6;
  
  thrust::adjacent_difference(thrust::cuda::par.on(s), input.begin(), input.end(), output.begin());

  cudaStreamSynchronize(s);
  
  ASSERT_EQUAL(output[0], 1);
  ASSERT_EQUAL(output[1], 3);
  ASSERT_EQUAL(output[2], 2);

  cudaStreamDestroy(s);
}
DECLARE_UNITTEST(TestAdjacentDifferenceCudaStreams);

