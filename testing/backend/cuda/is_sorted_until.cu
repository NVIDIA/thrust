#include <unittest/unittest.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>


template<typename ExecutionPolicy, typename Iterator1, typename Iterator2>
__global__
void is_sorted_until_kernel(ExecutionPolicy exec, Iterator1 first, Iterator1 last, Iterator2 result)
{
  *result = thrust::is_sorted_until(exec, first, last);
}


template<typename T, typename ExecutionPolicy>
void TestIsSortedUntilDevice(ExecutionPolicy exec, size_t n)
{
  n = thrust::max<size_t>(n, 2);

  thrust::device_vector<T> v = unittest::random_integers<T>(n);

  typedef typename thrust::device_vector<T>::iterator iter_type;

  thrust::device_vector<iter_type> result(1);

  v[0] = 1;
  v[1] = 0;
  
  is_sorted_until_kernel<<<1,1>>>(exec, v.begin(), v.end(), result.begin());
  ASSERT_EQUAL_QUIET(v.begin() + 1, (iter_type)result[0]);
  
  thrust::sort(v.begin(), v.end());
  
  is_sorted_until_kernel<<<1,1>>>(exec, v.begin(), v.end(), result.begin());
  ASSERT_EQUAL_QUIET(v.end(), (iter_type)result[0]);
}


template<typename T>
void TestIsSortedUntilDeviceSeq(const size_t n)
{
  TestIsSortedUntilDevice<T>(thrust::seq, n);
}
DECLARE_VARIABLE_UNITTEST(TestIsSortedUntilDeviceSeq);


template<typename T>
void TestIsSortedUntilDeviceDevice(const size_t n)
{
  TestIsSortedUntilDevice<T>(thrust::device, n);
}
DECLARE_VARIABLE_UNITTEST(TestIsSortedUntilDeviceDevice);

