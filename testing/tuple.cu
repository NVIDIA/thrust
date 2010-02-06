#include <thrusttest/unittest.h>
#include <thrust/tuple.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/generate.h>
#include <thrust/is_sorted.h>

using namespace thrusttest;
using namespace thrust;

template <typename T>
struct TestTupleConstructor
{
  void operator()(void)
  {
    host_vector<T> data = random_integers<T>(10);

    tuple<T> t1(data[0]);
    ASSERT_EQUAL(data[0], get<0>(t1));

    tuple<T,T> t2(data[0], data[1]);
    ASSERT_EQUAL(data[0], get<0>(t2));
    ASSERT_EQUAL(data[1], get<1>(t2));

    tuple<T,T,T> t3(data[0], data[1], data[2]);
    ASSERT_EQUAL(data[0], get<0>(t3));
    ASSERT_EQUAL(data[1], get<1>(t3));
    ASSERT_EQUAL(data[2], get<2>(t3));

    tuple<T,T,T,T> t4(data[0], data[1], data[2], data[3]);
    ASSERT_EQUAL(data[0], get<0>(t4));
    ASSERT_EQUAL(data[1], get<1>(t4));
    ASSERT_EQUAL(data[2], get<2>(t4));
    ASSERT_EQUAL(data[3], get<3>(t4));

    tuple<T,T,T,T,T> t5(data[0], data[1], data[2], data[3], data[4]);
    ASSERT_EQUAL(data[0], get<0>(t5));
    ASSERT_EQUAL(data[1], get<1>(t5));
    ASSERT_EQUAL(data[2], get<2>(t5));
    ASSERT_EQUAL(data[3], get<3>(t5));
    ASSERT_EQUAL(data[4], get<4>(t5));

    tuple<T,T,T,T,T,T> t6(data[0], data[1], data[2], data[3], data[4], data[5]);
    ASSERT_EQUAL(data[0], get<0>(t6));
    ASSERT_EQUAL(data[1], get<1>(t6));
    ASSERT_EQUAL(data[2], get<2>(t6));
    ASSERT_EQUAL(data[3], get<3>(t6));
    ASSERT_EQUAL(data[4], get<4>(t6));
    ASSERT_EQUAL(data[5], get<5>(t6));

    tuple<T,T,T,T,T,T,T> t7(data[0], data[1], data[2], data[3], data[4], data[5], data[6]);
    ASSERT_EQUAL(data[0], get<0>(t7));
    ASSERT_EQUAL(data[1], get<1>(t7));
    ASSERT_EQUAL(data[2], get<2>(t7));
    ASSERT_EQUAL(data[3], get<3>(t7));
    ASSERT_EQUAL(data[4], get<4>(t7));
    ASSERT_EQUAL(data[5], get<5>(t7));
    ASSERT_EQUAL(data[6], get<6>(t7));

    tuple<T,T,T,T,T,T,T,T> t8(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7]);
    ASSERT_EQUAL(data[0], get<0>(t8));
    ASSERT_EQUAL(data[1], get<1>(t8));
    ASSERT_EQUAL(data[2], get<2>(t8));
    ASSERT_EQUAL(data[3], get<3>(t8));
    ASSERT_EQUAL(data[4], get<4>(t8));
    ASSERT_EQUAL(data[5], get<5>(t8));
    ASSERT_EQUAL(data[6], get<6>(t8));
    ASSERT_EQUAL(data[7], get<7>(t8));

    tuple<T,T,T,T,T,T,T,T,T> t9(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8]);
    ASSERT_EQUAL(data[0], get<0>(t9));
    ASSERT_EQUAL(data[1], get<1>(t9));
    ASSERT_EQUAL(data[2], get<2>(t9));
    ASSERT_EQUAL(data[3], get<3>(t9));
    ASSERT_EQUAL(data[4], get<4>(t9));
    ASSERT_EQUAL(data[5], get<5>(t9));
    ASSERT_EQUAL(data[6], get<6>(t9));
    ASSERT_EQUAL(data[7], get<7>(t9));
    ASSERT_EQUAL(data[8], get<8>(t9));

    tuple<T,T,T,T,T,T,T,T,T,T> t10(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9]);
    ASSERT_EQUAL(data[0], get<0>(t10));
    ASSERT_EQUAL(data[1], get<1>(t10));
    ASSERT_EQUAL(data[2], get<2>(t10));
    ASSERT_EQUAL(data[3], get<3>(t10));
    ASSERT_EQUAL(data[4], get<4>(t10));
    ASSERT_EQUAL(data[5], get<5>(t10));
    ASSERT_EQUAL(data[6], get<6>(t10));
    ASSERT_EQUAL(data[7], get<7>(t10));
    ASSERT_EQUAL(data[8], get<8>(t10));
    ASSERT_EQUAL(data[9], get<9>(t10));
  }
};
SimpleUnitTest<TestTupleConstructor, NumericTypes> TestTupleConstructorInstance;

template <typename T>
struct TestMakeTuple
{
  void operator()(void)
  {
    host_vector<T> data = random_integers<T>(10);

    tuple<T> t1 = make_tuple(data[0]);
    ASSERT_EQUAL(data[0], get<0>(t1));

    tuple<T,T> t2 = make_tuple(data[0], data[1]);
    ASSERT_EQUAL(data[0], get<0>(t2));
    ASSERT_EQUAL(data[1], get<1>(t2));

    tuple<T,T,T> t3 = make_tuple(data[0], data[1], data[2]);
    ASSERT_EQUAL(data[0], get<0>(t3));
    ASSERT_EQUAL(data[1], get<1>(t3));
    ASSERT_EQUAL(data[2], get<2>(t3));

    tuple<T,T,T,T> t4 = make_tuple(data[0], data[1], data[2], data[3]);
    ASSERT_EQUAL(data[0], get<0>(t4));
    ASSERT_EQUAL(data[1], get<1>(t4));
    ASSERT_EQUAL(data[2], get<2>(t4));
    ASSERT_EQUAL(data[3], get<3>(t4));

    tuple<T,T,T,T,T> t5 = make_tuple(data[0], data[1], data[2], data[3], data[4]);
    ASSERT_EQUAL(data[0], get<0>(t5));
    ASSERT_EQUAL(data[1], get<1>(t5));
    ASSERT_EQUAL(data[2], get<2>(t5));
    ASSERT_EQUAL(data[3], get<3>(t5));
    ASSERT_EQUAL(data[4], get<4>(t5));

    tuple<T,T,T,T,T,T> t6 = make_tuple(data[0], data[1], data[2], data[3], data[4], data[5]);
    ASSERT_EQUAL(data[0], get<0>(t6));
    ASSERT_EQUAL(data[1], get<1>(t6));
    ASSERT_EQUAL(data[2], get<2>(t6));
    ASSERT_EQUAL(data[3], get<3>(t6));
    ASSERT_EQUAL(data[4], get<4>(t6));
    ASSERT_EQUAL(data[5], get<5>(t6));

    tuple<T,T,T,T,T,T,T> t7 = make_tuple(data[0], data[1], data[2], data[3], data[4], data[5], data[6]);
    ASSERT_EQUAL(data[0], get<0>(t7));
    ASSERT_EQUAL(data[1], get<1>(t7));
    ASSERT_EQUAL(data[2], get<2>(t7));
    ASSERT_EQUAL(data[3], get<3>(t7));
    ASSERT_EQUAL(data[4], get<4>(t7));
    ASSERT_EQUAL(data[5], get<5>(t7));
    ASSERT_EQUAL(data[6], get<6>(t7));

    tuple<T,T,T,T,T,T,T,T> t8 = make_tuple(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7]);
    ASSERT_EQUAL(data[0], get<0>(t8));
    ASSERT_EQUAL(data[1], get<1>(t8));
    ASSERT_EQUAL(data[2], get<2>(t8));
    ASSERT_EQUAL(data[3], get<3>(t8));
    ASSERT_EQUAL(data[4], get<4>(t8));
    ASSERT_EQUAL(data[5], get<5>(t8));
    ASSERT_EQUAL(data[6], get<6>(t8));
    ASSERT_EQUAL(data[7], get<7>(t8));

    tuple<T,T,T,T,T,T,T,T,T> t9 = make_tuple(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8]);
    ASSERT_EQUAL(data[0], get<0>(t9));
    ASSERT_EQUAL(data[1], get<1>(t9));
    ASSERT_EQUAL(data[2], get<2>(t9));
    ASSERT_EQUAL(data[3], get<3>(t9));
    ASSERT_EQUAL(data[4], get<4>(t9));
    ASSERT_EQUAL(data[5], get<5>(t9));
    ASSERT_EQUAL(data[6], get<6>(t9));
    ASSERT_EQUAL(data[7], get<7>(t9));
    ASSERT_EQUAL(data[8], get<8>(t9));

    tuple<T,T,T,T,T,T,T,T,T,T> t10 = make_tuple(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9]);
    ASSERT_EQUAL(data[0], get<0>(t10));
    ASSERT_EQUAL(data[1], get<1>(t10));
    ASSERT_EQUAL(data[2], get<2>(t10));
    ASSERT_EQUAL(data[3], get<3>(t10));
    ASSERT_EQUAL(data[4], get<4>(t10));
    ASSERT_EQUAL(data[5], get<5>(t10));
    ASSERT_EQUAL(data[6], get<6>(t10));
    ASSERT_EQUAL(data[7], get<7>(t10));
    ASSERT_EQUAL(data[8], get<8>(t10));
    ASSERT_EQUAL(data[9], get<9>(t10));
  }
};
SimpleUnitTest<TestMakeTuple, NumericTypes> TestMakeTupleInstance;

template <typename T>
struct TestTupleGet
{
  void operator()(void)
  {
    KNOWN_FAILURE
    //host_vector<T> data = random_integers<T>(10);

    //tuple<T> t1(data[0]);
    //ASSERT_EQUAL(data[0], t1.get<0>());

    //tuple<T,T> t2(data[0], data[1]);
    //ASSERT_EQUAL(data[0], t2.get<0>());
    //ASSERT_EQUAL(data[1], t2.get<1>());

    //tuple<T,T,T> t3 = make_tuple(data[0], data[1], data[2]);
    //ASSERT_EQUAL(data[0], t3.get<0>());
    //ASSERT_EQUAL(data[1], t3.get<1>());
    //ASSERT_EQUAL(data[2], t3.get<2>());

    //tuple<T,T,T,T> t4 = make_tuple(data[0], data[1], data[2], data[3]);
    //ASSERT_EQUAL(data[0], t4.get<0>());
    //ASSERT_EQUAL(data[1], t4.get<1>());
    //ASSERT_EQUAL(data[2], t4.get<2>());
    //ASSERT_EQUAL(data[3], t4.get<3>());

    //tuple<T,T,T,T,T> t5 = make_tuple(data[0], data[1], data[2], data[3], data[4]);
    //ASSERT_EQUAL(data[0], t5.get<0>());
    //ASSERT_EQUAL(data[1], t5.get<1>());
    //ASSERT_EQUAL(data[2], t5.get<2>());
    //ASSERT_EQUAL(data[3], t5.get<3>());
    //ASSERT_EQUAL(data[4], t5.get<4>());

    //tuple<T,T,T,T,T,T> t6 = make_tuple(data[0], data[1], data[2], data[3], data[4], data[5]);
    //ASSERT_EQUAL(data[0], t6.get<0>());
    //ASSERT_EQUAL(data[1], t6.get<1>());
    //ASSERT_EQUAL(data[2], t6.get<2>());
    //ASSERT_EQUAL(data[3], t6.get<3>());
    //ASSERT_EQUAL(data[4], t6.get<4>());
    //ASSERT_EQUAL(data[5], t6.get<5>());

    //tuple<T,T,T,T,T,T,T> t7 = make_tuple(data[0], data[1], data[2], data[3], data[4], data[5], data[6]);
    //ASSERT_EQUAL(data[0], t7.get<0>());
    //ASSERT_EQUAL(data[1], t7.get<1>());
    //ASSERT_EQUAL(data[2], t7.get<2>());
    //ASSERT_EQUAL(data[3], t7.get<3>());
    //ASSERT_EQUAL(data[4], t7.get<4>());
    //ASSERT_EQUAL(data[5], t7.get<5>());
    //ASSERT_EQUAL(data[6], t7.get<6>());

    //tuple<T,T,T,T,T,T,T,T> t8 = make_tuple(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7]);
    //ASSERT_EQUAL(data[0], t8.get<0>());
    //ASSERT_EQUAL(data[1], t8.get<1>());
    //ASSERT_EQUAL(data[2], t8.get<2>());
    //ASSERT_EQUAL(data[3], t8.get<3>());
    //ASSERT_EQUAL(data[4], t8.get<4>());
    //ASSERT_EQUAL(data[5], t8.get<5>());
    //ASSERT_EQUAL(data[6], t8.get<6>());
    //ASSERT_EQUAL(data[7], t8.get<7>());

    //tuple<T,T,T,T,T,T,T,T,T> t9 = make_tuple(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8]);
    //ASSERT_EQUAL(data[0], t9.get<0>());
    //ASSERT_EQUAL(data[1], t9.get<1>());
    //ASSERT_EQUAL(data[2], t9.get<2>());
    //ASSERT_EQUAL(data[3], t9.get<3>());
    //ASSERT_EQUAL(data[4], t9.get<4>());
    //ASSERT_EQUAL(data[5], t9.get<5>());
    //ASSERT_EQUAL(data[6], t9.get<6>());
    //ASSERT_EQUAL(data[7], t9.get<7>());
    //ASSERT_EQUAL(data[8], t9.get<8>());

    //tuple<T,T,T,T,T,T,T,T,T,T> t10 = make_tuple(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9]);
    //ASSERT_EQUAL(data[0], t10.get<0>());
    //ASSERT_EQUAL(data[1], t10.get<1>());
    //ASSERT_EQUAL(data[2], t10.get<2>());
    //ASSERT_EQUAL(data[3], t10.get<3>());
    //ASSERT_EQUAL(data[4], t10.get<4>());
    //ASSERT_EQUAL(data[5], t10.get<5>());
    //ASSERT_EQUAL(data[6], t10.get<6>());
    //ASSERT_EQUAL(data[7], t10.get<7>());
    //ASSERT_EQUAL(data[8], t10.get<8>());
    //ASSERT_EQUAL(data[9], t10.get<9>());
  }
};
SimpleUnitTest<TestTupleGet, NumericTypes> TestTupleGetInstance;



template <typename T>
struct TestTupleComparison
{
  void operator()(void)
  {
    tuple<T,T,T,T,T> lhs(0, 0, 0, 0, 0), rhs(0, 0, 0, 0, 0);

    // equality
    ASSERT_EQUAL(true,  lhs == rhs);
    get<0>(rhs) = 1;
    ASSERT_EQUAL(false,  lhs == rhs);

    // inequality
    ASSERT_EQUAL(true,  lhs != rhs);
    lhs = rhs;
    ASSERT_EQUAL(false, lhs != rhs);

    // less than
    lhs = make_tuple(0,0,0,0,0);
    rhs = make_tuple(0,0,1,0,0);
    ASSERT_EQUAL(true,  lhs < rhs);
    get<0>(lhs) = 2;
    ASSERT_EQUAL(false, lhs < rhs);

    // less than equal
    lhs = make_tuple(0,0,0,0,0);
    rhs = lhs;
    ASSERT_EQUAL(true,  lhs <= rhs); // equal
    get<2>(rhs) = 1;
    ASSERT_EQUAL(true,  lhs <= rhs); // less than
    get<2>(lhs) = 2;
    ASSERT_EQUAL(false, lhs <= rhs);

    // greater than
    lhs = make_tuple(1,0,0,0,0);
    rhs = make_tuple(0,1,1,1,1);
    ASSERT_EQUAL(true,  lhs > rhs);
    get<0>(rhs) = 2;
    ASSERT_EQUAL(false, lhs > rhs);

    // greater than equal
    lhs = make_tuple(0,0,0,0,0);
    rhs = lhs;
    ASSERT_EQUAL(true,  lhs >= rhs); // equal
    get<4>(lhs) = 1;
    ASSERT_EQUAL(true,  lhs >= rhs); // greater than
    get<3>(rhs) = 1;
    ASSERT_EQUAL(false, lhs >= rhs);
  }
};
SimpleUnitTest<TestTupleComparison, NumericTypes> TestTupleComparisonInstance;

template<typename Tuple>
__host__ __device__
Tuple operator+(const Tuple &lhs, const Tuple &rhs)
{
  return make_tuple(get<0>(lhs) + get<0>(rhs),
                    get<1>(lhs) + get<1>(rhs));
}

struct SortTuplesByFirst
{
  template<typename T1, typename T2>
  __host__ __device__
  bool operator()(T1 &lhs, T2 &rhs)
  {
    return get<0>(lhs) < get<0>(rhs);
  }
};

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
     thrust::host_vector<T> h_keys   = thrusttest::random_integers<T>(n);
     thrust::host_vector<T> h_values = thrusttest::random_integers<T>(n);

     // zip up the data
     thrust::host_vector< tuple<T,T> > h_tuples(n);
     thrust::transform(h_keys.begin(),   h_keys.end(),
                        h_values.begin(), h_tuples.begin(),
                        MakeTupleFunctor());

     // copy to device
     thrust::device_vector< tuple<T,T> > d_tuples = h_tuples;

     // sort on host
     thrust::stable_sort(h_tuples.begin(), h_tuples.end(), SortTuplesByFirst());

     // sort on device
     thrust::stable_sort(d_tuples.begin(), d_tuples.end(), SortTuplesByFirst());

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

template <typename T>
struct TestTupleReduce
{
  void operator()(const size_t n)
  {
     thrust::host_vector<T> h_t1 = thrusttest::random_integers<T>(n);
     thrust::host_vector<T> h_t2 = thrusttest::random_integers<T>(n);

     // zip up the data
     thrust::host_vector< tuple<T,T> > h_tuples(n);
     thrust::transform(h_t1.begin(), h_t1.end(),
                        h_t2.begin(), h_tuples.begin(),
                        MakeTupleFunctor());

     // copy to device
     thrust::device_vector< tuple<T,T> > d_tuples = h_tuples;

     // sum on host
     tuple<T,T> h_result = thrust::reduce(h_tuples.begin(), h_tuples.end());

     // sum on device
     tuple<T,T> d_result = thrust::reduce(d_tuples.begin(), d_tuples.end());

     ASSERT_EQUAL_QUIET(h_result, d_result);
  }
};
VariableUnitTest<TestTupleReduce, IntegralTypes> TestTupleReduceInstance;

template <typename T>
struct TestTupleScan
{
  void operator()(const size_t n)
  {
     thrust::host_vector<T> h_t1 = thrusttest::random_integers<T>(n);
     thrust::host_vector<T> h_t2 = thrusttest::random_integers<T>(n);

     // zip up the data
     thrust::host_vector< tuple<T,T> > h_tuples(n);
     thrust::transform(h_t1.begin(), h_t1.end(),
                       h_t2.begin(), h_tuples.begin(),
                       MakeTupleFunctor());

     // copy to device
     thrust::device_vector< tuple<T,T> > d_tuples = h_tuples;

     tuple<T,T> zero(0,0);

     // scan on host
     thrust::exclusive_scan(h_tuples.begin(), h_tuples.begin(), h_tuples.begin(), zero);

     // scan on device
     thrust::exclusive_scan(d_tuples.begin(), d_tuples.begin(), d_tuples.begin(), zero);

     ASSERT_EQUAL_QUIET(h_tuples, d_tuples);
  }
};
VariableUnitTest<TestTupleScan, IntegralTypes> TestTupleScanInstance;


template <typename T>
struct TestTupleTransform
{
  void operator()(const size_t n)
  {
     thrust::host_vector<T> h_t1 = thrusttest::random_integers<T>(n);
     thrust::host_vector<T> h_t2 = thrusttest::random_integers<T>(n);

     // zip up the data
     thrust::host_vector< tuple<T,T> > h_tuples(n);
     thrust::transform(h_t1.begin(), h_t1.end(),
                       h_t2.begin(), h_tuples.begin(),
                       MakeTupleFunctor());

     // copy to device
     thrust::device_vector< tuple<T,T> > d_tuples = h_tuples;

     thrust::device_vector<T> d_t1(n), d_t2(n);

     // select 0th
     thrust::transform(d_tuples.begin(), d_tuples.end(), d_t1.begin(), GetFunctor<0>());

     // select 1st
     thrust::transform(d_tuples.begin(), d_tuples.end(), d_t2.begin(), GetFunctor<1>());

     ASSERT_ALMOST_EQUAL(h_t1, d_t1);
     ASSERT_ALMOST_EQUAL(h_t2, d_t2);

     ASSERT_EQUAL_QUIET(h_tuples, d_tuples);
  }
};
VariableUnitTest<TestTupleTransform, NumericTypes> TestTupleTransformInstance;

template <typename T>
struct TestTupleTieFunctor
{
  __host__ __device__
  void clear(T *data) const
  {
    for(int i = 0; i < 10; ++i)
      data[i] = 13;
  }

  __host__ __device__
  bool operator()() const
  {
    using namespace thrust;

    bool result = true;

    T data[10];
    clear(data);

    tie(data[0]) = make_tuple(0);;
    result &= data[0] == 0;
    clear(data);

    tie(data[0], data[1]) = make_tuple(0,1);
    result &= data[0] == 0;
    result &= data[1] == 1;
    clear(data);

    tie(data[0], data[1], data[2]) = make_tuple(0,1,2);
    result &= data[0] == 0;
    result &= data[1] == 1;
    result &= data[2] == 2;
    clear(data);

    tie(data[0], data[1], data[2], data[3]) = make_tuple(0,1,2,3);
    result &= data[0] == 0;
    result &= data[1] == 1;
    result &= data[2] == 2;
    result &= data[3] == 3;
    clear(data);

    tie(data[0], data[1], data[2], data[3], data[4]) = make_tuple(0,1,2,3,4);
    result &= data[0] == 0;
    result &= data[1] == 1;
    result &= data[2] == 2;
    result &= data[3] == 3;
    result &= data[4] == 4;
    clear(data);

    tie(data[0], data[1], data[2], data[3], data[4], data[5]) = make_tuple(0,1,2,3,4,5);
    result &= data[0] == 0;
    result &= data[1] == 1;
    result &= data[2] == 2;
    result &= data[3] == 3;
    result &= data[4] == 4;
    result &= data[5] == 5;
    clear(data);

    tie(data[0], data[1], data[2], data[3], data[4], data[5], data[6]) = make_tuple(0,1,2,3,4,5,6);
    result &= data[0] == 0;
    result &= data[1] == 1;
    result &= data[2] == 2;
    result &= data[3] == 3;
    result &= data[4] == 4;
    result &= data[5] == 5;
    result &= data[6] == 6;
    clear(data);

    tie(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7]) = make_tuple(0,1,2,3,4,5,6,7);
    result &= data[0] == 0;
    result &= data[1] == 1;
    result &= data[2] == 2;
    result &= data[3] == 3;
    result &= data[4] == 4;
    result &= data[5] == 5;
    result &= data[6] == 6;
    result &= data[7] == 7;
    clear(data);

    tie(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8]) = make_tuple(0,1,2,3,4,5,6,7,8);
    result &= data[0] == 0;
    result &= data[1] == 1;
    result &= data[2] == 2;
    result &= data[3] == 3;
    result &= data[4] == 4;
    result &= data[5] == 5;
    result &= data[6] == 6;
    result &= data[7] == 7;
    result &= data[8] == 8;
    clear(data);

    tie(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9]) = make_tuple(0,1,2,3,4,5,6,7,8,9);
    result &= data[0] == 0;
    result &= data[1] == 1;
    result &= data[2] == 2;
    result &= data[3] == 3;
    result &= data[4] == 4;
    result &= data[5] == 5;
    result &= data[6] == 6;
    result &= data[7] == 7;
    result &= data[8] == 8;
    result &= data[9] == 9;
    clear(data);

    return result;
  }
};

template <typename T>
struct TestTupleTie
{
  void operator()(void)
  {
    thrust::host_vector<bool> h_result(1);
    thrust::generate(h_result.begin(), h_result.end(), TestTupleTieFunctor<T>());

    thrust::device_vector<bool> d_result(1);
    thrust::generate(d_result.begin(), d_result.end(), TestTupleTieFunctor<T>());

    ASSERT_EQUAL(true, h_result[0]);
    ASSERT_EQUAL(true, d_result[0]);
  }
};
SimpleUnitTest<TestTupleTie, NumericTypes> TestTupleTieInstance;

