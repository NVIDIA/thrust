#include <unittest/unittest.h>
#include <thrust/partition.h>
#include <thrust/functional.h>

template<typename T>
struct is_even
{
  __host__ __device__
  bool operator()(T x) const { return ((int) x % 2) == 0; }
};

template<typename Vector>
void TestPartitionPointSimple(void)
{
  typedef typename Vector::value_type T;
  typedef typename Vector::iterator Iterator;

  Vector v(4);
  v[0] = 1; v[1] = 1; v[2] = 1; v[3] = 0;

  Iterator first = v.begin();

  Iterator last = v.begin() + 4;
  Iterator ref = first + 3;
  ASSERT_EQUAL_QUIET(ref, thrust::partition_point(first, last, thrust::identity<T>()));

  last = v.begin() + 3;
  ref = last;
  ASSERT_EQUAL_QUIET(ref, thrust::partition_point(first, last, thrust::identity<T>()));
}
DECLARE_VECTOR_UNITTEST(TestPartitionPointSimple);

template <class Vector>
void TestPartitionPoint(void)
{
#if defined(CUDA_VERSION) && (CUDA_VERSION == 3020)
  KNOWN_FAILURE;
#else
  typedef typename Vector::value_type T;
  typedef typename Vector::iterator Iterator;

  const size_t n = (1 << 16) + 13;

  Vector v = unittest::random_integers<T>(n);

  Iterator ref = thrust::stable_partition(v.begin(), v.end(), is_even<T>());

  ASSERT_EQUAL(ref - v.begin(), thrust::partition_point(v.begin(), v.end(), is_even<T>()) - v.begin());
#endif
}
DECLARE_VECTOR_UNITTEST(TestPartitionPoint);


template<typename ForwardIterator, typename Predicate>
ForwardIterator partition_point(my_system &system, 
                                ForwardIterator first,
                                ForwardIterator last,
                                Predicate pred)
{
  system.validate_dispatch();
  return first;
}

void TestPartitionPointDispatchExplicit()
{
  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::partition_point(sys,
                          vec.begin(),
                          vec.begin(),
                          0);

  ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestPartitionPointDispatchExplicit);


template<typename ForwardIterator, typename Predicate>
ForwardIterator partition_point(my_tag,
                                ForwardIterator first,
                                ForwardIterator last,
                                Predicate pred)
{
  *first = 13;
  return first;
}

void TestPartitionPointDispatchImplicit()
{
  thrust::device_vector<int> vec(1);

  thrust::partition_point(thrust::retag<my_tag>(vec.begin()),
                          thrust::retag<my_tag>(vec.begin()),
                          0);

  ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestPartitionPointDispatchImplicit);

