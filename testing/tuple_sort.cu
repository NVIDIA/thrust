#include <unittest/unittest.h>
#include <thrust/tuple.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/is_sorted.h>

using namespace unittest;
using namespace thrust;

struct MakeTupleFunctor
{
  template<typename T1, typename T2>
  __host__ __device__
  tuple<T1,T2> operator()(T1 &lhs, T2 &rhs)
  {
    return make_tuple(lhs, rhs);
  }
};

template<int N>
struct GetFunctor
{
  template<typename Tuple>
  __host__ __device__
  typename access_traits<
                    typename tuple_element<N, Tuple>::type
                  >::const_type
  operator()(const Tuple &t)
  {
    return get<N>(t);
  }
};

template <typename T>
struct TestTupleStableSort
{
  void operator()(const size_t n)
  {
     thrust::host_vector<T> h_keys   = unittest::random_integers<T>(n);
     thrust::host_vector<T> h_values = unittest::random_integers<T>(n);

     // zip up the data
     thrust::host_vector< tuple<T,T> > h_tuples(n);
     thrust::transform(h_keys.begin(),   h_keys.end(),
                        h_values.begin(), h_tuples.begin(),
                        MakeTupleFunctor());

     // copy to device
     thrust::device_vector< tuple<T,T> > d_tuples = h_tuples;

     // sort on host
     thrust::stable_sort(h_tuples.begin(), h_tuples.end());

     // sort on device
     thrust::stable_sort(d_tuples.begin(), d_tuples.end());

     ASSERT_EQUAL(true, thrust::is_sorted(d_tuples.begin(), d_tuples.end()));

     // select keys
     thrust::transform(h_tuples.begin(), h_tuples.end(), h_keys.begin(), GetFunctor<0>());

     thrust::device_vector<T> d_keys(h_keys.size());
     thrust::transform(d_tuples.begin(), d_tuples.end(), d_keys.begin(), GetFunctor<0>());

     // select values
     thrust::transform(h_tuples.begin(), h_tuples.end(), h_values.begin(), GetFunctor<1>());
     
     thrust::device_vector<T> d_values(h_values.size());
     thrust::transform(d_tuples.begin(), d_tuples.end(), d_values.begin(), GetFunctor<1>());

     ASSERT_ALMOST_EQUAL(h_keys, d_keys);
     ASSERT_ALMOST_EQUAL(h_values, d_values);
  }
};
VariableUnitTest<TestTupleStableSort, NumericTypes> TestTupleStableSortInstance;

