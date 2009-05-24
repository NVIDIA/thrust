#include <thrusttest/unittest.h>
#include <thrust/generate.h>

template<typename T>
struct return_value
{
    T val;

    return_value(void){}
    return_value(T v):val(v){}

    __host__ __device__
    T operator()(void){ return val; }
};

template<class Vector>
void TestGenerateSimple(void)
{
    typedef typename Vector::value_type T;

    Vector result(5);

    T value = 13;

    return_value<T> f(value);

    thrust::generate(result.begin(), result.end(), f);

    ASSERT_EQUAL(result[0], value);
    ASSERT_EQUAL(result[1], value);
    ASSERT_EQUAL(result[2], value);
    ASSERT_EQUAL(result[3], value);
    ASSERT_EQUAL(result[4], value);
}
DECLARE_VECTOR_UNITTEST(TestGenerateSimple);

template <typename T>
void TestGenerate(const size_t n)
{
    thrust::host_vector<T> h_result(n);
    thrust::device_vector<T> d_result(n);

    T value = 13;
    return_value<T> f(value);

    thrust::generate(h_result.begin(), h_result.end(), f);
    thrust::generate(d_result.begin(), d_result.end(), f);

    ASSERT_EQUAL(h_result, d_result);
}
DECLARE_VARIABLE_UNITTEST(TestGenerate);

