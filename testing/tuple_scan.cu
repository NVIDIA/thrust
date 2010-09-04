#include <unittest/unittest.h>
#include <thrust/tuple.h>
#include <thrust/scan.h>
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
struct TestTupleScan
{
  void operator()(const size_t n)
  {
     thrust::host_vector<T> h_t1 = unittest::random_integers<T>(n);
     thrust::host_vector<T> h_t2 = unittest::random_integers<T>(n);

     // initialize input
     thrust::host_vector< tuple<T,T> > h_input(n);
     thrust::transform(h_t1.begin(), h_t1.end(),
                       h_t2.begin(), h_input.begin(),
                       MakeTupleFunctor());
     thrust::device_vector< tuple<T,T> > d_input = h_input;
     
     // allocate output
     tuple<T,T> zero(0,0);
     thrust::host_vector  < tuple<T,T> > h_output(n, zero);
     thrust::device_vector< tuple<T,T> > d_output(n, zero);

     // exclusive_scan
     thrust::inclusive_scan(h_input.begin(), h_input.end(), h_output.begin());
     thrust::inclusive_scan(d_input.begin(), d_input.end(), d_output.begin());
     ASSERT_EQUAL_QUIET(h_output, d_output);

     // exclusive_scan
     tuple<T,T> init(13,17);
     thrust::exclusive_scan(h_input.begin(), h_input.end(), h_output.begin(), init);
     thrust::exclusive_scan(d_input.begin(), d_input.end(), d_output.begin(), init);

     ASSERT_EQUAL_QUIET(h_output, d_output);
  }
};
VariableUnitTest<TestTupleScan, IntegralTypes> TestTupleScanInstance;

