#include <thrusttest/unittest.h>
#include <thrust/find.h>

template <typename T>
struct equal_to_value_pred
{
    T value;

    equal_to_value_pred(T value) : value(value) {}

    __host__ __device__
    bool operator()(T v) const { return v == value; }
};

template <class Vector>
void TestFindSimple(void)
{
    typedef typename Vector::value_type T;

    Vector vec(5);
    vec[0] = 1;
    vec[1] = 2;
    vec[2] = 3;
    vec[3] = 4;
    vec[4] = 5;

    ASSERT_EQUAL(thrust::find(vec.begin(), vec.end(), 0) - vec.begin(), 5);
    ASSERT_EQUAL(thrust::find(vec.begin(), vec.end(), 1) - vec.begin(), 0);
    ASSERT_EQUAL(thrust::find(vec.begin(), vec.end(), 2) - vec.begin(), 1);
    ASSERT_EQUAL(thrust::find(vec.begin(), vec.end(), 3) - vec.begin(), 2);
    ASSERT_EQUAL(thrust::find(vec.begin(), vec.end(), 4) - vec.begin(), 3);
    ASSERT_EQUAL(thrust::find(vec.begin(), vec.end(), 5) - vec.begin(), 4);
    ASSERT_EQUAL(thrust::find(vec.begin(), vec.end(), 6) - vec.begin(), 5);
}
DECLARE_VECTOR_UNITTEST(TestFindSimple);

template <class Vector>
void TestFindIfSimple(void)
{
    typedef typename Vector::value_type T;

    Vector vec(5);
    vec[0] = 1;
    vec[1] = 2;
    vec[2] = 3;
    vec[3] = 4;
    vec[4] = 5;

    ASSERT_EQUAL(thrust::find_if(vec.begin(), vec.end(), equal_to_value_pred<T>(0)) - vec.begin(), 5);
    ASSERT_EQUAL(thrust::find_if(vec.begin(), vec.end(), equal_to_value_pred<T>(1)) - vec.begin(), 0);
    ASSERT_EQUAL(thrust::find_if(vec.begin(), vec.end(), equal_to_value_pred<T>(2)) - vec.begin(), 1);
    ASSERT_EQUAL(thrust::find_if(vec.begin(), vec.end(), equal_to_value_pred<T>(3)) - vec.begin(), 2);
    ASSERT_EQUAL(thrust::find_if(vec.begin(), vec.end(), equal_to_value_pred<T>(4)) - vec.begin(), 3);
    ASSERT_EQUAL(thrust::find_if(vec.begin(), vec.end(), equal_to_value_pred<T>(5)) - vec.begin(), 4);
    ASSERT_EQUAL(thrust::find_if(vec.begin(), vec.end(), equal_to_value_pred<T>(6)) - vec.begin(), 5);
}
DECLARE_VECTOR_UNITTEST(TestFindIfSimple);


