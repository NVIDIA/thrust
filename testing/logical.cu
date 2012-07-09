#include <unittest/unittest.h>
#include <thrust/logical.h>
#include <thrust/functional.h>

struct my_system : thrust::device_system<my_system> {};

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


template <class InputIterator, class Predicate>
bool all_of(my_system, InputIterator first, InputIterator last, Predicate pred)
{
    *first = 13;
    return false;
}

void TestAllOfDispatchExplicit()
{
    thrust::device_vector<int> vec(1);

    my_system sys;
    thrust::all_of(sys, vec.begin(), vec.end(), 0);

    ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestAllOfDispatchExplicit);

void TestAllOfDispatchImplicit()
{
    thrust::device_vector<int> vec(1);

    thrust::all_of(thrust::retag<my_system>(vec.begin()),
                   thrust::retag<my_system>(vec.end()),
                   0);

    ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestAllOfDispatchImplicit);


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


template <class InputIterator, class Predicate>
bool any_of(my_system, InputIterator first, InputIterator last, Predicate pred)
{
    *first = 13;
    return false;
}

void TestAnyOfDispatchExplicit()
{
    thrust::device_vector<int> vec(1);

    my_system sys;
    thrust::any_of(sys, vec.begin(), vec.end(), 0);

    ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestAnyOfDispatchExplicit);

void TestAnyOfDispatchImplicit()
{
    thrust::device_vector<int> vec(1);

    thrust::any_of(thrust::retag<my_system>(vec.begin()),
                   thrust::retag<my_system>(vec.end()),
                   0);

    ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestAnyOfDispatchImplicit);


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


template <class InputIterator, class Predicate>
bool none_of(my_system, InputIterator first, InputIterator last, Predicate pred)
{
    *first = 13;
    return false;
}

void TestNoneOfDispatchExplicit()
{
    thrust::device_vector<int> vec(1);

    my_system sys;
    thrust::none_of(sys, vec.begin(), vec.end(), 0);

    ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestNoneOfDispatchExplicit);

void TestNoneOfDispatchImplicit()
{
    thrust::device_vector<int> vec(1);

    thrust::none_of(thrust::retag<my_system>(vec.begin()),
                    thrust::retag<my_system>(vec.end()),
                    0);

    ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestNoneOfDispatchImplicit);

