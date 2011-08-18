#include <unittest/unittest.h>

#include <thrust/functional.h>
#include <thrust/detail/backend/decompose.h>
#include <thrust/detail/backend/reduce_intervals.h>

using thrust::detail::backend::uniform_decomposition;
using thrust::detail::backend::reduce_intervals;

template <class Vector>
void TestReduceIntervalsSimple(void)
{
  typedef typename Vector::value_type T;

  Vector input(10, 1);
    
  {
    uniform_decomposition<int> decomp(10, 10, 1);
    Vector output(decomp.size());
    reduce_intervals(input.begin(), output.begin(), thrust::plus<T>(), decomp);

    ASSERT_EQUAL(output[0], 10);
  }
  
  {
    uniform_decomposition<int> decomp(10, 6, 2);
    Vector output(decomp.size());
    reduce_intervals(input.begin(), output.begin(), thrust::plus<T>(), decomp);

    ASSERT_EQUAL(output[0], 6);
    ASSERT_EQUAL(output[1], 4);
  }
}
DECLARE_VECTOR_UNITTEST(TestReduceIntervalsSimple);


template <typename T>
struct TestReduceIntervals
{
  void operator()(const size_t n)
  {
    thrust::host_vector<T>   h_input = unittest::random_integers<T>(n);
    thrust::device_vector<T> d_input = h_input;

    uniform_decomposition<size_t> decomp(n, 7, 100);

    thrust::host_vector<T>   h_output(decomp.size());
    thrust::device_vector<T> d_output(decomp.size());
    
    reduce_intervals(h_input.begin(), h_output.begin(), thrust::plus<T>(), decomp);
    reduce_intervals(d_input.begin(), d_output.begin(), thrust::plus<T>(), decomp);

    ASSERT_EQUAL(h_output, d_output);
  }
};
VariableUnitTest<TestReduceIntervals, IntegralTypes> TestReduceIntervalsInstance;

