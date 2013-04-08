#include <unittest/unittest.h>
#include <thrust/partition.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>


template<typename Iterator, typename Predicate, typename Iterator2>
__global__
void is_partitioned_kernel(Iterator first, Iterator last, Predicate pred, Iterator2 result)
{
  *result = thrust::is_partitioned(thrust::seq, first, last, pred);
}


template<typename T>
struct is_even
{
  __host__ __device__
  bool operator()(T x) const { return ((int) x % 2) == 0; }
};


template<typename T>
void TestIsPartitionedDeviceSeq(size_t n)
{
  n = thrust::max<size_t>(n, 2);

  thrust::device_vector<T> v = unittest::random_integers<T>(n);

  thrust::device_vector<bool> result(1);

  v[0] = 1;
  v[1] = 0;

  is_partitioned_kernel<<<1,1>>>(v.begin(), v.end(), is_even<T>(), result.begin());

  ASSERT_EQUAL(false, result[0]);

  thrust::partition(v.begin(), v.end(), is_even<T>());

  is_partitioned_kernel<<<1,1>>>(v.begin(), v.end(), is_even<T>(), result.begin());

  ASSERT_EQUAL(true, result[0]);
}
DECLARE_VARIABLE_UNITTEST(TestIsPartitionedDeviceSeq);

