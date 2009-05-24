#include <thrusttest/unittest.h>
#include <thrust/reduce.h>

template <class Vector>
void TestReduceSimple(void)
{
    typedef typename Vector::value_type T;

    Vector v(3);
    v[0] = 1; v[1] = -2; v[2] = 3;

    // no initializer
    ASSERT_EQUAL(thrust::reduce(v.begin(), v.end()), 2);

    // with initializer
    ASSERT_EQUAL(thrust::reduce(v.begin(), v.end(), (T) 10), 12);
}
DECLARE_VECTOR_UNITTEST(TestReduceSimple);


template <typename T>
void TestReduce(const size_t n)
{
    thrust::host_vector<T>   h_data = thrusttest::random_integers<T>(n);
    thrust::device_vector<T> d_data = h_data;

    T init = 13;

    T cpu_result = thrust::reduce(h_data.begin(), h_data.end(), init);
    T gpu_result = thrust::reduce(d_data.begin(), d_data.end(), init);

    ASSERT_ALMOST_EQUAL(cpu_result, gpu_result);
}
DECLARE_VARIABLE_UNITTEST(TestReduce);


template <class IntVector, class FloatVector>
void TestReduceMixedTypes(void)
{
    // make sure we get types for default args and operators correct
    IntVector int_input(4);
    int_input[0] = 1;
    int_input[1] = 2;
    int_input[2] = 3;
    int_input[3] = 4;

    FloatVector float_input(4);
    float_input[0] = 1.5;
    float_input[1] = 2.5;
    float_input[2] = 3.5;
    float_input[3] = 4.5;

    // float -> int should use using plus<int> operator by default
    ASSERT_EQUAL(thrust::reduce(float_input.begin(), float_input.end(), (int) 0), 10);
    
    // int -> float should use using plus<float> operator by default
    ASSERT_EQUAL(thrust::reduce(int_input.begin(), int_input.end(), (float) 0.5), 10.5);
}
void TestReduceMixedTypesHost(void)
{
    TestReduceMixedTypes< thrust::host_vector<int>, thrust::host_vector<float> >();
}
DECLARE_UNITTEST(TestReduceMixedTypesHost);
void TestReduceMixedTypesDevice(void)
{
    TestReduceMixedTypes< thrust::device_vector<int>, thrust::device_vector<float> >();
}
DECLARE_UNITTEST(TestReduceMixedTypesDevice);



template <typename T>
void TestReduceWithOperator(const size_t n)
{
    thrust::host_vector<T>   h_data = thrusttest::random_integers<T>(n);
    thrust::device_vector<T> d_data = h_data;

    T init = 0;

    T cpu_result = thrust::reduce(h_data.begin(), h_data.end(), init, thrust::maximum<T>());
    T gpu_result = thrust::reduce(d_data.begin(), d_data.end(), init, thrust::maximum<T>());

    ASSERT_EQUAL(cpu_result, gpu_result);
}
DECLARE_VARIABLE_UNITTEST(TestReduceWithOperator);



template <typename T>
struct TestReducePair
{
  void operator()(const size_t n)
  {
#ifdef __APPLE__
    // test_pair<char,char> fails on OSX
    KNOWN_FAILURE
#else
    thrusttest::random_integer<T> rnd;

    thrust::host_vector< thrusttest::test_pair<T,T> > h_data(n);
    for(size_t i = 0; i < n; i++){
        h_data[i].first  = rnd();
        h_data[i].second = rnd();
    }
    thrust::device_vector< thrusttest::test_pair<T,T> > d_data = h_data;

    thrusttest::test_pair<T,T> init;
    init.first = 13;
    init.second = 29;

    thrusttest::test_pair<T,T> cpu_result = thrust::reduce(h_data.begin(), h_data.end(), init);
    thrusttest::test_pair<T,T> gpu_result = thrust::reduce(d_data.begin(), d_data.end(), init);

    ASSERT_EQUAL(cpu_result, gpu_result);
#endif // __APPLE__
  }
};
VariableUnitTest<TestReducePair, IntegralTypes> TestReducePairInstance;

