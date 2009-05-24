#include <thrusttest/unittest.h>
#include <thrust/transform_reduce.h>

template <class Vector>
void TestTransformReduceSimple(void)
{
    typedef typename Vector::value_type T;

    Vector data(3);
    data[0] = 1; data[1] = -2; data[2] = 3;

    T init = 10;
    T result = thrust::transform_reduce(data.begin(), data.end(), thrust::negate<T>(), init, thrust::plus<T>());

    ASSERT_EQUAL(result, 8);
}
DECLARE_VECTOR_UNITTEST(TestTransformReduceSimple);

template <typename T>
void TestTransformReduce(const size_t n)
{
    thrust::host_vector<T>   h_data = thrusttest::random_integers<T>(n);
    thrust::device_vector<T> d_data = h_data;

    T init = 13;

    T cpu_result = thrust::transform_reduce(h_data.begin(), h_data.end(), thrust::negate<T>(), init, thrust::plus<T>());
    T gpu_result = thrust::transform_reduce(d_data.begin(), d_data.end(), thrust::negate<T>(), init, thrust::plus<T>());

    ASSERT_ALMOST_EQUAL(cpu_result, gpu_result);
}
DECLARE_VARIABLE_UNITTEST(TestTransformReduce);

template <typename T>
void TestTransformReduceFromConst(const size_t n)
{
    thrust::host_vector<T>   h_data = thrusttest::random_integers<T>(n);
    thrust::device_vector<T> d_data = h_data;

    T init = 13;

    T cpu_result = thrust::transform_reduce(h_data.cbegin(), h_data.cend(), thrust::negate<T>(), init, thrust::plus<T>());
    T gpu_result = thrust::transform_reduce(d_data.cbegin(), d_data.cend(), thrust::negate<T>(), init, thrust::plus<T>());

    ASSERT_ALMOST_EQUAL(cpu_result, gpu_result);
}
DECLARE_VARIABLE_UNITTEST(TestTransformReduceFromConst);

