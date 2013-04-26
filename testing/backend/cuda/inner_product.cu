#include <unittest/unittest.h>
#include <thrust/inner_product.h>
#include <thrust/execution_policy.h>


template<typename ExecutionPolicy, typename Iterator1, typename Iterator2, typename T, typename Iterator3>
__global__
void inner_product_kernel(ExecutionPolicy exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, T init, Iterator3 result)
{
  *result = thrust::inner_product(exec, first1, last1, first2, init);
}


template<typename T, typename ExecutionPolicy>
void TestInnerProductDevice(ExecutionPolicy exec, const size_t n)
{
  thrust::host_vector<T> h_v1 = unittest::random_integers<T>(n);
  thrust::host_vector<T> h_v2 = unittest::random_integers<T>(n);
  
  thrust::device_vector<T> d_v1 = h_v1;
  thrust::device_vector<T> d_v2 = h_v2;
  
  thrust::device_vector<T> result(1);
  
  T init = 13;
  
  T expected = thrust::inner_product(h_v1.begin(), h_v1.end(), h_v2.begin(), init);
  inner_product_kernel<<<1,1>>>(exec, d_v1.begin(), d_v1.end(), d_v2.begin(), init, result.begin());
  
  ASSERT_EQUAL(expected, result[0]);
}


template<typename T>
struct TestInnerProductDeviceSeq
{
  void operator()(const size_t n)
  {
    TestInnerProductDevice<T>(thrust::seq, n);
  }
};
VariableUnitTest<TestInnerProductDeviceSeq, IntegralTypes> TestInnerProductDeviceSeqInstance;


template<typename T>
struct TestInnerProductDeviceDevice
{
  void operator()(const size_t n)
  {
    TestInnerProductDevice<T>(thrust::device, n);
  }
};
VariableUnitTest<TestInnerProductDeviceDevice, IntegralTypes> TestInnerProductDeviceDeviceInstance;

