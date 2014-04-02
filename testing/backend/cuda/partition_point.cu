#include <unittest/unittest.h>
#include <thrust/partition.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>


template<typename Iterator1, typename Predicate, typename Iterator2>
__global__
void partition_point_kernel(Iterator1 first, Iterator1 last, Predicate pred, Iterator2 result)
{
  *result = thrust::partition_point(thrust::seq, first, last, pred);
}


template<typename T>
struct is_even
{
  __host__ __device__
  bool operator()(T x) const { return ((int) x % 2) == 0; }
};


template<typename T>
void TestPartitionPointDeviceSeq(size_t n)
{
  thrust::device_vector<T> v = unittest::random_integers<T>(n);
  typedef typename thrust::device_vector<T>::iterator iterator;

  iterator ref = thrust::stable_partition(v.begin(), v.end(), is_even<T>());

  thrust::device_vector<iterator> result(1);
  partition_point_kernel<<<1,1>>>(v.begin(), v.end(), is_even<T>(), result.begin());

  ASSERT_EQUAL(ref - v.begin(), (iterator)result[0] - v.begin());
}
DECLARE_VARIABLE_UNITTEST(TestPartitionPointDeviceSeq);


void TestPartitionPointCudaStreams()
{
  typedef thrust::device_vector<int> Vector;
  typedef typename Vector::value_type T;
  typedef typename Vector::iterator Iterator;

  Vector v(4);
  v[0] = 1; v[1] = 1; v[2] = 1; v[3] = 0;

  Iterator first = v.begin();

  Iterator last = v.begin() + 4;
  Iterator ref = first + 3;

  cudaStream_t s;
  cudaStreamCreate(&s);

  ASSERT_EQUAL_QUIET(ref, thrust::partition_point(thrust::cuda::par(s), first, last, thrust::identity<T>()));

  last = v.begin() + 3;
  ref = last;
  ASSERT_EQUAL_QUIET(ref, thrust::partition_point(thrust::cuda::par(s), first, last, thrust::identity<T>()));

  cudaStreamDestroy(s);
}
DECLARE_UNITTEST(TestPartitionPointCudaStreams);

