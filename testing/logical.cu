#include <unittest/unittest.h>
#include <thrust/logical.h>
#include <thrust/functional.h>

template <class Vector>
void TestAllOf(void)
{
    typedef typename Vector::value_type T;

    Vector v(3, 1);

    ASSERT_EQUAL(thrust::all_of(v.begin(), v.end(), thrust::identity<T>()), true);

    v[1] = 0;
    
    ASSERT_EQUAL(thrust::all_of(v.begin(), v.end(), thrust::identity<T>()), false);

    ASSERT_EQUAL(thrust::all_of(v.begin() + 0, v.begin() + 0, thrust::identity<T>()), true);
    ASSERT_EQUAL(thrust::all_of(v.begin() + 0, v.begin() + 1, thrust::identity<T>()), true);
    ASSERT_EQUAL(thrust::all_of(v.begin() + 0, v.begin() + 2, thrust::identity<T>()), false);
    ASSERT_EQUAL(thrust::all_of(v.begin() + 1, v.begin() + 2, thrust::identity<T>()), false);
}
DECLARE_VECTOR_UNITTEST(TestAllOf);


template <class Vector>
void TestAnyOf(void)
{
    typedef typename Vector::value_type T;

    Vector v(3, 1);

    ASSERT_EQUAL(thrust::any_of(v.begin(), v.end(), thrust::identity<T>()), true);

    v[1] = 0;
    
    ASSERT_EQUAL(thrust::any_of(v.begin(), v.end(), thrust::identity<T>()), true);

    ASSERT_EQUAL(thrust::any_of(v.begin() + 0, v.begin() + 0, thrust::identity<T>()), false);
    ASSERT_EQUAL(thrust::any_of(v.begin() + 0, v.begin() + 1, thrust::identity<T>()), true);
    ASSERT_EQUAL(thrust::any_of(v.begin() + 0, v.begin() + 2, thrust::identity<T>()), true);
    ASSERT_EQUAL(thrust::any_of(v.begin() + 1, v.begin() + 2, thrust::identity<T>()), false);
}
DECLARE_VECTOR_UNITTEST(TestAnyOf);


template <class Vector>
void TestNoneOf(void)
{
    typedef typename Vector::value_type T;

    Vector v(3, 1);

    ASSERT_EQUAL(thrust::none_of(v.begin(), v.end(), thrust::identity<T>()), false);

    v[1] = 0;
    
    ASSERT_EQUAL(thrust::none_of(v.begin(), v.end(), thrust::identity<T>()), false);

    ASSERT_EQUAL(thrust::none_of(v.begin() + 0, v.begin() + 0, thrust::identity<T>()), true);
    ASSERT_EQUAL(thrust::none_of(v.begin() + 0, v.begin() + 1, thrust::identity<T>()), false);
    ASSERT_EQUAL(thrust::none_of(v.begin() + 0, v.begin() + 2, thrust::identity<T>()), false);
    ASSERT_EQUAL(thrust::none_of(v.begin() + 1, v.begin() + 2, thrust::identity<T>()), true);
}
DECLARE_VECTOR_UNITTEST(TestNoneOf);

