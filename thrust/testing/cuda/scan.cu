#include <cstdio>
#include <unittest/unittest.h>
#include <thrust/scan.h>
#include <thrust/functional.h>


#ifdef THRUST_TEST_DEVICE_SIDE
template<typename ExecutionPolicy, typename Iterator1, typename Iterator2>
__global__
void inclusive_scan_kernel(ExecutionPolicy exec, Iterator1 first, Iterator1 last, Iterator2 result)
{
  thrust::inclusive_scan(exec, first, last, result);
}


template<typename ExecutionPolicy, typename Iterator1, typename Iterator2>
__global__
void exclusive_scan_kernel(ExecutionPolicy exec, Iterator1 first, Iterator1 last, Iterator2 result)
{
  thrust::exclusive_scan(exec, first, last, result);
}


template<typename ExecutionPolicy, typename Iterator1, typename Iterator2, typename T>
__global__
void exclusive_scan_kernel(ExecutionPolicy exec, Iterator1 first, Iterator1 last, Iterator2 result, T init)
{
  thrust::exclusive_scan(exec, first, last, result, init);
}


template<typename T, typename ExecutionPolicy>
void TestScanDevice(ExecutionPolicy exec, const size_t n)
{
  thrust::host_vector<T>   h_input = unittest::random_integers<T>(n);
  thrust::device_vector<T> d_input = h_input;
  
  thrust::host_vector<T>   h_output(n);
  thrust::device_vector<T> d_output(n);
  
  thrust::inclusive_scan(h_input.begin(), h_input.end(), h_output.begin());

  inclusive_scan_kernel<<<1,1>>>(exec, d_input.begin(), d_input.end(), d_output.begin());
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }

  ASSERT_EQUAL(d_output, h_output);
  
  thrust::exclusive_scan(h_input.begin(), h_input.end(), h_output.begin());

  exclusive_scan_kernel<<<1,1>>>(exec, d_input.begin(), d_input.end(), d_output.begin());
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }

  ASSERT_EQUAL(d_output, h_output);
  
  thrust::exclusive_scan(h_input.begin(), h_input.end(), h_output.begin(), (T) 11);

  exclusive_scan_kernel<<<1,1>>>(exec, d_input.begin(), d_input.end(), d_output.begin(), (T) 11);
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }

  ASSERT_EQUAL(d_output, h_output);
  
  // in-place scans
  h_output = h_input;
  d_output = d_input;

  thrust::inclusive_scan(h_output.begin(), h_output.end(), h_output.begin());

  inclusive_scan_kernel<<<1,1>>>(exec, d_output.begin(), d_output.end(), d_output.begin());
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }

  ASSERT_EQUAL(d_output, h_output);
  
  h_output = h_input;
  d_output = d_input;
  
  thrust::exclusive_scan(h_output.begin(), h_output.end(), h_output.begin());

  exclusive_scan_kernel<<<1,1>>>(exec, d_output.begin(), d_output.end(), d_output.begin());
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }
  
  ASSERT_EQUAL(d_output, h_output);
}


template<typename T>
struct TestScanDeviceSeq
{
  void operator()(const size_t n)
  {
    TestScanDevice<T>(thrust::seq, n);
  }
};
VariableUnitTest<TestScanDeviceSeq, IntegralTypes> TestScanDeviceSeqInstance;


template<typename T>
struct TestScanDeviceDevice
{
  void operator()(const size_t n)
  {
    TestScanDevice<T>(thrust::device, n);
  }
};
VariableUnitTest<TestScanDeviceDevice, IntegralTypes> TestScanDeviceDeviceInstance;
#endif


void TestScanCudaStreams()
{
  typedef thrust::device_vector<int> Vector;
  typedef Vector::value_type T;
  
  Vector::iterator iter;

  Vector input(5);
  Vector result(5);
  Vector output(5);

  input[0] = 1; input[1] = 3; input[2] = -2; input[3] = 4; input[4] = -5;

  Vector input_copy(input);

  cudaStream_t s;
  cudaStreamCreate(&s);

  // inclusive scan
  iter = thrust::inclusive_scan(thrust::cuda::par.on(s), input.begin(), input.end(), output.begin());
  cudaStreamSynchronize(s);

  result[0] = 1; result[1] = 4; result[2] = 2; result[3] = 6; result[4] = 1;
  ASSERT_EQUAL(std::size_t(iter - output.begin()), input.size());
  ASSERT_EQUAL(input,  input_copy);
  ASSERT_EQUAL(output, result);
  
  // exclusive scan
  iter = thrust::exclusive_scan(thrust::cuda::par.on(s), input.begin(), input.end(), output.begin(), 0);
  cudaStreamSynchronize(s);

  result[0] = 0; result[1] = 1; result[2] = 4; result[3] = 2; result[4] = 6;
  ASSERT_EQUAL(std::size_t(iter - output.begin()), input.size());
  ASSERT_EQUAL(input,  input_copy);
  ASSERT_EQUAL(output, result);
  
  // exclusive scan with init
  iter = thrust::exclusive_scan(thrust::cuda::par.on(s), input.begin(), input.end(), output.begin(), 3);
  cudaStreamSynchronize(s);

  result[0] = 3; result[1] = 4; result[2] = 7; result[3] = 5; result[4] = 9;
  ASSERT_EQUAL(std::size_t(iter - output.begin()), input.size());
  ASSERT_EQUAL(input,  input_copy);
  ASSERT_EQUAL(output, result);
  
  // inclusive scan with op
  iter = thrust::inclusive_scan(thrust::cuda::par.on(s), input.begin(), input.end(), output.begin(), thrust::plus<T>());
  cudaStreamSynchronize(s);

  result[0] = 1; result[1] = 4; result[2] = 2; result[3] = 6; result[4] = 1;
  ASSERT_EQUAL(std::size_t(iter - output.begin()), input.size());
  ASSERT_EQUAL(input,  input_copy);
  ASSERT_EQUAL(output, result);

  // exclusive scan with init and op
  iter = thrust::exclusive_scan(thrust::cuda::par.on(s), input.begin(), input.end(), output.begin(), 3, thrust::plus<T>());
  cudaStreamSynchronize(s);

  result[0] = 3; result[1] = 4; result[2] = 7; result[3] = 5; result[4] = 9;
  ASSERT_EQUAL(std::size_t(iter - output.begin()), input.size());
  ASSERT_EQUAL(input,  input_copy);
  ASSERT_EQUAL(output, result);

  // inplace inclusive scan
  input = input_copy;
  iter = thrust::inclusive_scan(thrust::cuda::par.on(s), input.begin(), input.end(), input.begin());
  cudaStreamSynchronize(s);

  result[0] = 1; result[1] = 4; result[2] = 2; result[3] = 6; result[4] = 1;
  ASSERT_EQUAL(std::size_t(iter - input.begin()), input.size());
  ASSERT_EQUAL(input, result);

  // inplace exclusive scan with init
  input = input_copy;
  iter = thrust::exclusive_scan(thrust::cuda::par.on(s), input.begin(), input.end(), input.begin(), 3);
  cudaStreamSynchronize(s);

  result[0] = 3; result[1] = 4; result[2] = 7; result[3] = 5; result[4] = 9;
  ASSERT_EQUAL(std::size_t(iter - input.begin()), input.size());
  ASSERT_EQUAL(input, result);

  // inplace exclusive scan with implicit init=0
  input = input_copy;
  iter = thrust::exclusive_scan(thrust::cuda::par.on(s), input.begin(), input.end(), input.begin());
  cudaStreamSynchronize(s);

  result[0] = 0; result[1] = 1; result[2] = 4; result[3] = 2; result[4] = 6;
  ASSERT_EQUAL(std::size_t(iter - input.begin()), input.size());
  ASSERT_EQUAL(input, result);

  cudaStreamDestroy(s);
}
DECLARE_UNITTEST(TestScanCudaStreams);

template <typename T>
struct const_ref_plus_mod3
{
    T * table;

    const_ref_plus_mod3(T * table) : table(table) {}

    __host__ __device__
    const T& operator()(T a, T b)
    {
        return table[(int) (a + b)];
    }
};

static void TestInclusiveScanWithConstAccumulator(void)
{
    // add numbers modulo 3 with external lookup table
    thrust::device_vector<int> data(7);
    data[0] = 0;
    data[1] = 1;
    data[2] = 2;
    data[3] = 1;
    data[4] = 2;
    data[5] = 0;
    data[6] = 1;

    thrust::device_vector<int> table(6);
    table[0] = 0;
    table[1] = 1;
    table[2] = 2;
    table[3] = 0;
    table[4] = 1;
    table[5] = 2;

    thrust::inclusive_scan(data.begin(), data.end(), data.begin(), const_ref_plus_mod3<int>(thrust::raw_pointer_cast(&table[0])));
    
    ASSERT_EQUAL(data[0], 0);
    ASSERT_EQUAL(data[1], 1);
    ASSERT_EQUAL(data[2], 0);
    ASSERT_EQUAL(data[3], 1);
    ASSERT_EQUAL(data[4], 0);
    ASSERT_EQUAL(data[5], 0);
    ASSERT_EQUAL(data[6], 1);
}
DECLARE_UNITTEST(TestInclusiveScanWithConstAccumulator);
