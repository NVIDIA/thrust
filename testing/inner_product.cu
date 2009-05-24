#include <thrusttest/unittest.h>
#include <thrust/inner_product.h>

template <class Vector>
void TestInnerProductSimple(void)
{
    typedef typename Vector::value_type T;

    Vector v1(3);
    Vector v2(3);
    v1[0] =  1; v1[1] = -2; v1[2] =  3;
    v2[0] = -4; v2[1] =  5; v2[2] =  6;

    T init = 3;
    T result = thrust::inner_product(v1.begin(), v1.end(), v2.begin(), init);
    ASSERT_EQUAL(result, 7);
}
DECLARE_VECTOR_UNITTEST(TestInnerProductSimple);

template <class Vector>
void TestInnerProductWithOperator(void)
{
    typedef typename Vector::value_type T;

    Vector v1(3);
    Vector v2(3);
    v1[0] =  1; v1[1] = -2; v1[2] =  3;
    v2[0] = -1; v2[1] =  3; v2[2] =  6;

    // compute (v1 - v2) and perform a multiplies reduction
    T init = 3;
    T result = thrust::inner_product(v1.begin(), v1.end(), v2.begin(), init, 
                                      thrust::multiplies<T>(), thrust::minus<T>());
    ASSERT_EQUAL(result, 90);
}
DECLARE_VECTOR_UNITTEST(TestInnerProductWithOperator);

template <typename T>
void TestInnerProduct(const size_t n)
{
    thrust::host_vector<T> h_v1 = thrusttest::random_integers<T>(n);
    thrust::host_vector<T> h_v2 = thrusttest::random_integers<T>(n);

    thrust::device_vector<T> d_v1 = h_v1;
    thrust::device_vector<T> d_v2 = h_v2;

    T init = 13;

    T cpu_result = thrust::inner_product(h_v1.begin(), h_v1.end(), h_v2.begin(), init);
    T gpu_result = thrust::inner_product(d_v1.begin(), d_v1.end(), d_v2.begin(), init);

    ASSERT_ALMOST_EQUAL(cpu_result, gpu_result);
}
DECLARE_VARIABLE_UNITTEST(TestInnerProduct);


