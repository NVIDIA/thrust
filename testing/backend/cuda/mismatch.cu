#include <unittest/unittest.h>
#include <thrust/mismatch.h>
#include <thrust/execution_policy.h>


template<typename ExecutionPolicy, typename Iterator1, typename Iterator2, typename Iterator3>
__global__ void mismatch_kernel(ExecutionPolicy exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator3 result)
{
  *result = thrust::mismatch(exec, first1, last1, first2);
}


template<typename ExecutionPolicy>
void TestMismatchDevice(ExecutionPolicy exec)
{
  thrust::device_vector<int> a(4);
  thrust::device_vector<int> b(4);
  a[0] = 1; b[0] = 1;
  a[1] = 2; b[1] = 2;
  a[2] = 3; b[2] = 4;
  a[3] = 4; b[3] = 3;

  typedef thrust::pair<
    typename thrust::device_vector<int>::iterator,
    typename thrust::device_vector<int>::iterator
  > pair_type;

  thrust::device_vector<pair_type> d_result(1);
  
  mismatch_kernel<<<1,1>>>(exec, a.begin(), a.end(), b.begin(), d_result.begin());

  ASSERT_EQUAL(2, ((pair_type)d_result[0]).first  - a.begin());
  ASSERT_EQUAL(2, ((pair_type)d_result[0]).second - b.begin());
  
  b[2] = 3;
  
  mismatch_kernel<<<1,1>>>(exec, a.begin(), a.end(), b.begin(), d_result.begin());
  ASSERT_EQUAL(3, ((pair_type)d_result[0]).first  - a.begin());
  ASSERT_EQUAL(3, ((pair_type)d_result[0]).second - b.begin());
  
  b[3] = 4;
  
  mismatch_kernel<<<1,1>>>(exec, a.begin(), a.end(), b.begin(), d_result.begin());
  ASSERT_EQUAL(4, ((pair_type)d_result[0]).first  - a.begin());
  ASSERT_EQUAL(4, ((pair_type)d_result[0]).second - b.begin());
}


void TestMismatchDeviceSeq()
{
  TestMismatchDevice(thrust::seq);
}
DECLARE_UNITTEST(TestMismatchDeviceSeq);


void TestMismatchDeviceDevice()
{
  TestMismatchDevice(thrust::device);
}
DECLARE_UNITTEST(TestMismatchDeviceDevice);

