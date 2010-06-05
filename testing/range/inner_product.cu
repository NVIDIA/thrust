#include <unittest/unittest.h>
#include <thrust/range/algorithm/inner_product.h>

template <class Vector>
void TestRangeInnerProductSimple(void)
{
    typedef typename Vector::value_type T;

    Vector v1(3);
    Vector v2(3);
    v1[0] =  1; v1[1] = -2; v1[2] =  3;
    v2[0] = -4; v2[1] =  5; v2[2] =  6;

    using namespace thrust::experimental::range;

    T init = 3;
    T result = inner_product(v1, v2, init);
    ASSERT_EQUAL(7, result);
}
DECLARE_VECTOR_UNITTEST(TestRangeInnerProductSimple);

template <class Vector>
void TestRangeInnerProductWithOperator(void)
{
    typedef typename Vector::value_type T;

    Vector v1(3);
    Vector v2(3);
    v1[0] =  1; v1[1] = -2; v1[2] =  3;
    v2[0] = -1; v2[1] =  3; v2[2] =  6;

    using namespace thrust::experimental::range;

    // compute (v1 - v2) and perform a multiplies reduction
    T init = 3;
    T result = inner_product(v1, v2, init, 
                             thrust::multiplies<T>(), thrust::minus<T>());
    ASSERT_EQUAL(90, result);
}
DECLARE_VECTOR_UNITTEST(TestRangeInnerProductWithOperator);

template <typename T>
void TestRangeInnerProduct(const size_t n)
{
    thrust::host_vector<T> h_v1 = unittest::random_integers<T>(n);
    thrust::host_vector<T> h_v2 = unittest::random_integers<T>(n);

    thrust::device_vector<T> d_v1 = h_v1;
    thrust::device_vector<T> d_v2 = h_v2;
    
    using namespace thrust::experimental::range;

    T init = 13;

    T cpu_result = inner_product(h_v1, h_v2, init);
    T gpu_result = inner_product(d_v1, d_v2, init);

    ASSERT_ALMOST_EQUAL(cpu_result, gpu_result);
}
DECLARE_VARIABLE_UNITTEST(TestRangeInnerProduct);


