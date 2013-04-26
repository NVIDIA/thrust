#include <unittest/unittest.h>
#include <thrust/equal.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>


template<typename ExecutionPolicy, typename Iterator1, typename Iterator2, typename Iterator3>
__global__
void equal_kernel(ExecutionPolicy exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator3 result)
{
  *result = thrust::equal(exec, first1, last1, first2);
}


template<typename ExecutionPolicy, typename Iterator1, typename Iterator2, typename BinaryPredicate, typename Iterator3>
__global__
void equal_kernel(ExecutionPolicy exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, BinaryPredicate pred, Iterator3 result)
{
  *result = thrust::equal(exec, first1, last1, first2, pred);
}


template<typename T, typename ExecutionPolicy>
void TestEqualDevice(ExecutionPolicy exec, const size_t n)
{
  thrust::device_vector<T> d_data1 = unittest::random_samples<T>(n);
  thrust::device_vector<T> d_data2 = unittest::random_samples<T>(n);
  thrust::device_vector<bool> d_result(1, false);
  
  //empty ranges
  equal_kernel<<<1,1>>>(exec, d_data1.begin(), d_data1.begin(), d_data1.begin(), d_result.begin());
  ASSERT_EQUAL(d_result[0], true);
  
  //symmetric cases
  equal_kernel<<<1,1>>>(exec, d_data1.begin(), d_data1.end(), d_data1.begin(), d_result.begin());
  ASSERT_EQUAL(d_result[0], true);
  
  if(n > 0)
  {
    d_data1[0] = 0; d_data2[0] = 1;
    
    //different vectors
    equal_kernel<<<1,1>>>(exec, d_data1.begin(), d_data1.end(), d_data2.begin(), d_result.begin());
    ASSERT_EQUAL(d_result[0], false);
    
    //different predicates
    equal_kernel<<<1,1>>>(exec, d_data1.begin(), d_data1.begin() + 1, d_data2.begin(), thrust::less<T>(), d_result.begin());
    ASSERT_EQUAL(d_result[0], true);
    equal_kernel<<<1,1>>>(exec, d_data1.begin(), d_data1.begin() + 1, d_data2.begin(), thrust::greater<T>(), d_result.begin());
    ASSERT_EQUAL(d_result[0], false);
  }
}


template<typename T>
void TestEqualDeviceSeq(const size_t n)
{
  TestEqualDevice<T>(thrust::seq, n);
}
DECLARE_VARIABLE_UNITTEST(TestEqualDeviceSeq);


template<typename T>
void TestEqualDeviceDevice(const size_t n)
{
  TestEqualDevice<T>(thrust::device, n);
}
DECLARE_VARIABLE_UNITTEST(TestEqualDeviceDevice);

