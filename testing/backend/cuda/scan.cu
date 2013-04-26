#include <cstdio>
#include <unittest/unittest.h>
#include <thrust/scan.h>
#include <thrust/functional.h>


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
  ASSERT_EQUAL(d_output, h_output);
  
  thrust::exclusive_scan(h_input.begin(), h_input.end(), h_output.begin());
  exclusive_scan_kernel<<<1,1>>>(exec, d_input.begin(), d_input.end(), d_output.begin());
  ASSERT_EQUAL(d_output, h_output);
  
  thrust::exclusive_scan(h_input.begin(), h_input.end(), h_output.begin(), (T) 11);
  exclusive_scan_kernel<<<1,1>>>(exec, d_input.begin(), d_input.end(), d_output.begin(), (T) 11);
  ASSERT_EQUAL(d_output, h_output);
  
  // in-place scans
  h_output = h_input;
  d_output = d_input;
  thrust::inclusive_scan(h_output.begin(), h_output.end(), h_output.begin());
  inclusive_scan_kernel<<<1,1>>>(exec, d_output.begin(), d_output.end(), d_output.begin());
  ASSERT_EQUAL(d_output, h_output);
  
#if CUDA_VERSION > 5000
  h_output = h_input;
  d_output = d_input;
  
  thrust::exclusive_scan(h_output.begin(), h_output.end(), h_output.begin());
  exclusive_scan_kernel<<<1,1>>>(exec, d_output.begin(), d_output.end(), d_output.begin());
  
  ASSERT_EQUAL(d_output, h_output);
#else
  KNOWN_FAILURE; // XXX nvcc 5 generates bad code for inplace sequential exclusive_scan
#endif
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

