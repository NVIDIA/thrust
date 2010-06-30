#include <unittest/unittest.h>
#include <thrust/tuple.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>

using namespace unittest;
using namespace thrust;

template <typename Tuple>
__host__ __device__
Tuple operator+(const Tuple &lhs, const Tuple &rhs)
{
  return make_tuple(get<0>(lhs) + get<0>(rhs),
                    get<1>(lhs) + get<1>(rhs));
}

struct MakeTupleFunctor
{
  template<typename T1, typename T2>
  __host__ __device__
  tuple<T1,T2> operator()(T1 &lhs, T2 &rhs)
  {
    return make_tuple(lhs, rhs);
  }
};

template <typename T>
struct TestTupleReduce
{
  void operator()(const size_t n)
  {
     thrust::host_vector<T> h_t1 = unittest::random_integers<T>(n);
     thrust::host_vector<T> h_t2 = unittest::random_integers<T>(n);

     // zip up the data
     thrust::host_vector< tuple<T,T> > h_tuples(n);
     thrust::transform(h_t1.begin(), h_t1.end(),
                        h_t2.begin(), h_tuples.begin(),
                        MakeTupleFunctor());

     // copy to device
     thrust::device_vector< tuple<T,T> > d_tuples = h_tuples;
     
     tuple<T,T> zero(0,0);

     // sum on host
     tuple<T,T> h_result = thrust::reduce(h_tuples.begin(), h_tuples.end(), zero);

     // sum on device
     tuple<T,T> d_result = thrust::reduce(d_tuples.begin(), d_tuples.end(), zero);

     ASSERT_EQUAL_QUIET(h_result, d_result);
  }
};
VariableUnitTest<TestTupleReduce, IntegralTypes> TestTupleReduceInstance;

