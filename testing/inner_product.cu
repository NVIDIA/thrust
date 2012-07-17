#include <unittest/unittest.h>
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

struct my_system : thrust::device_system<my_system> {};

template <typename InputIterator1, typename InputIterator2, typename OutputType>
int inner_product(my_system, InputIterator1, InputIterator1, InputIterator2, OutputType)
{
    return 13;
}

void TestInnerProductDispatchExplicit()
{
    thrust::device_vector<int> vec;

    my_system sys;
    int result = thrust::inner_product(sys,
                                       vec.begin(),
                                       vec.end(),
                                       vec.begin(),
                                       0);

    ASSERT_EQUAL(13, result);
}
DECLARE_UNITTEST(TestInnerProductDispatchExplicit);

void TestInnerProductDispatchImplicit()
{
    thrust::device_vector<int> vec;

    int result = thrust::inner_product(thrust::retag<my_system>(vec.begin()),
                                       thrust::retag<my_system>(vec.end()),
                                       thrust::retag<my_system>(vec.begin()),
                                       0);

    ASSERT_EQUAL(13, result);
}
DECLARE_UNITTEST(TestInnerProductDispatchImplicit);

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
struct TestInnerProduct
{
    void operator()(const size_t n)
    {
        thrust::host_vector<T> h_v1 = unittest::random_integers<T>(n);
        thrust::host_vector<T> h_v2 = unittest::random_integers<T>(n);

        thrust::device_vector<T> d_v1 = h_v1;
        thrust::device_vector<T> d_v2 = h_v2;

        T init = 13;

        T expected = thrust::inner_product(h_v1.begin(), h_v1.end(), h_v2.begin(), init);
        T result   = thrust::inner_product(d_v1.begin(), d_v1.end(), d_v2.begin(), init);

        ASSERT_EQUAL(expected, result);
    }
};
VariableUnitTest<TestInnerProduct, IntegralTypes> TestInnerProductInstance;


