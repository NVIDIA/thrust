#include <unittest/unittest.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>


template<typename Iterator1, typename Iterator2>
__global__
void is_sorted_until_kernel(Iterator1 first, Iterator1 last, Iterator2 result)
{
  *result = thrust::is_sorted_until(thrust::seq, first, last);
}


template<typename T>
void TestIsSortedUntilDeviceSeq(size_t n)
{
  n = thrust::max<size_t>(n, 2);

  thrust::device_vector<T> v = unittest::random_integers<T>(n);

  typedef typename thrust::device_vector<T>::iterator iter_type;

  thrust::device_vector<iter_type> result(1);

  v[0] = 1;
  v[1] = 0;
  
  is_sorted_until_kernel<<<1,1>>>(v.begin(), v.end(), result.begin());
  ASSERT_EQUAL_QUIET(v.begin() + 1, (iter_type)result[0]);
  
  thrust::sort(v.begin(), v.end());
  
  is_sorted_until_kernel<<<1,1>>>(v.begin(), v.end(), result.begin());
  ASSERT_EQUAL_QUIET(v.end(), (iter_type)result[0]);
}
DECLARE_VARIABLE_UNITTEST(TestIsSortedUntilDeviceSeq);

