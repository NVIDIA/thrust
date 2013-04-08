#include <unittest/unittest.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>


template<typename Iterator, typename Iterator2>
__global__
void is_sorted_kernel(Iterator first, Iterator last, Iterator2 result)
{
  *result = thrust::is_sorted(thrust::seq, first, last);
}


template<typename T>
void TestIsSortedDeviceSeq(size_t n)
{
  n = thrust::max<size_t>(n, 2);

  thrust::device_vector<T> v = unittest::random_integers<T>(n);

  thrust::device_vector<bool> result(1);

  v[0] = 1;
  v[1] = 0;

  is_sorted_kernel<<<1,1>>>(v.begin(), v.end(), result.begin());
  ASSERT_EQUAL(false, result[0]);

  thrust::sort(v.begin(), v.end());

  is_sorted_kernel<<<1,1>>>(v.begin(), v.end(), result.begin());
  ASSERT_EQUAL(true, result[0]);
}
DECLARE_VARIABLE_UNITTEST(TestIsSortedDeviceSeq);

