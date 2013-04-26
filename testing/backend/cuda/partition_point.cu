#include <unittest/unittest.h>
#include <thrust/partition.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>


template<typename ExecutionPolicy, typename Iterator1, typename Predicate, typename Iterator2>
__global__
void partition_point_kernel(ExecutionPolicy exec, Iterator1 first, Iterator1 last, Predicate pred, Iterator2 result)
{
  *result = thrust::partition_point(exec, first, last, pred);
}


template<typename T>
struct is_even
{
  __host__ __device__
  bool operator()(T x) const { return ((int) x % 2) == 0; }
};


template<typename T, typename ExecutionPolicy>
void TestPartitionPointDevice(ExecutionPolicy exec, size_t n)
{
  thrust::device_vector<T> v = unittest::random_integers<T>(n);
  typedef typename thrust::device_vector<T>::iterator iterator;

  iterator ref = thrust::stable_partition(v.begin(), v.end(), is_even<T>());

  thrust::device_vector<iterator> result(1);
  partition_point_kernel<<<1,1>>>(exec, v.begin(), v.end(), is_even<T>(), result.begin());

  ASSERT_EQUAL(ref - v.begin(), (iterator)result[0] - v.begin());
}


template<typename T>
void TestPartitionPointDeviceSeq(const size_t n)
{
  TestPartitionPointDevice<T>(thrust::seq, n);
}
DECLARE_VARIABLE_UNITTEST(TestPartitionPointDeviceSeq);


template<typename T>
void TestPartitionPointDeviceDevice(const size_t n)
{
  TestPartitionPointDevice<T>(thrust::device, n);
}
DECLARE_VARIABLE_UNITTEST(TestPartitionPointDeviceDevice);


