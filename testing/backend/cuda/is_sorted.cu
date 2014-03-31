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


void TestIsSortedCudaStreams()
{
  thrust::device_vector<int> v(4);
  v[0] = 0; v[1] = 5; v[2] = 8; v[3] = 0;

  cudaStream_t s;
  cudaStreamCreate(&s);
  
  ASSERT_EQUAL(thrust::is_sorted(thrust::cuda::par(s), v.begin(), v.begin() + 0), true);
  ASSERT_EQUAL(thrust::is_sorted(thrust::cuda::par(s), v.begin(), v.begin() + 1), true);
  
  // the following line crashes gcc 4.3
#if (__GNUC__ == 4) && (__GNUC_MINOR__ == 3)
  // do nothing
#else
  // compile this line on other compilers
  ASSERT_EQUAL(thrust::is_sorted(thrust::cuda::par(s), v.begin(), v.begin() + 2), true);
#endif // GCC

  ASSERT_EQUAL(thrust::is_sorted(thrust::cuda::par(s), v.begin(), v.begin() + 3), true);
  ASSERT_EQUAL(thrust::is_sorted(thrust::cuda::par(s), v.begin(), v.begin() + 4), false);
  
  ASSERT_EQUAL(thrust::is_sorted(thrust::cuda::par(s), v.begin(), v.begin() + 3, thrust::less<int>()),    true);
  
  ASSERT_EQUAL(thrust::is_sorted(thrust::cuda::par(s), v.begin(), v.begin() + 1, thrust::greater<int>()), true);
  ASSERT_EQUAL(thrust::is_sorted(thrust::cuda::par(s), v.begin(), v.begin() + 4, thrust::greater<int>()), false);
  
  ASSERT_EQUAL(thrust::is_sorted(thrust::cuda::par(s), v.begin(), v.end()), false);

  cudaStreamDestroy(s);
}
DECLARE_UNITTEST(TestIsSortedCudaStreams);

