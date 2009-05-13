#include <komradetest/unittest.h>
#include <komrade/transform_reduce.h>

template <class Vector>
void TestTransformReduceSimple(void)
{
    typedef typename Vector::value_type T;

    Vector data(3);
    data[0] = 1; data[1] = -2; data[2] = 3;

    T init = 10;
    T result = komrade::transform_reduce(data.begin(), data.end(), komrade::negate<T>(), init, komrade::plus<T>());

    ASSERT_EQUAL(result, 8);
}
DECLARE_VECTOR_UNITTEST(TestTransformReduceSimple);

template <typename T>
void TestTransformReduce(const size_t n)
{
    komrade::host_vector<T>   h_data = komradetest::random_integers<T>(n);
    komrade::device_vector<T> d_data = h_data;

    T init = 13;

    T cpu_result = komrade::transform_reduce(h_data.begin(), h_data.end(), komrade::negate<T>(), init, komrade::plus<T>());
    T gpu_result = komrade::transform_reduce(d_data.begin(), d_data.end(), komrade::negate<T>(), init, komrade::plus<T>());

    ASSERT_ALMOST_EQUAL(cpu_result, gpu_result);
}
DECLARE_VARIABLE_UNITTEST(TestTransformReduce);

template <typename T>
void TestTransformReduceFromConst(const size_t n)
{
    komrade::host_vector<T>   h_data = komradetest::random_integers<T>(n);
    komrade::device_vector<T> d_data = h_data;

    T init = 13;

    T cpu_result = komrade::transform_reduce(h_data.cbegin(), h_data.cend(), komrade::negate<T>(), init, komrade::plus<T>());
    T gpu_result = komrade::transform_reduce(d_data.cbegin(), d_data.cend(), komrade::negate<T>(), init, komrade::plus<T>());

    ASSERT_ALMOST_EQUAL(cpu_result, gpu_result);
}
DECLARE_VARIABLE_UNITTEST(TestTransformReduceFromConst);

