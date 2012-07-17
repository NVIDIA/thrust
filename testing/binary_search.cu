#include <unittest/unittest.h>
#include <thrust/binary_search.h>

#include <thrust/sequence.h>
#include <thrust/sort.h>

__THRUST_DISABLE_MSVC_POSSIBLE_LOSS_OF_DATA_WARNING_BEGIN

// for testing dispatch
struct my_system : thrust::device_system<my_system> {};

//////////////////////
// Scalar Functions //
//////////////////////

template <class Vector>
void TestScalarLowerBoundSimple(void)
{
    typedef typename Vector::value_type T;

    Vector vec(5);

    vec[0] = 0;
    vec[1] = 2;
    vec[2] = 5;
    vec[3] = 7;
    vec[4] = 8;

    ASSERT_EQUAL(thrust::lower_bound(vec.begin(), vec.end(), 0) - vec.begin(), 0);
    ASSERT_EQUAL(thrust::lower_bound(vec.begin(), vec.end(), 1) - vec.begin(), 1);
    ASSERT_EQUAL(thrust::lower_bound(vec.begin(), vec.end(), 2) - vec.begin(), 1);
    ASSERT_EQUAL(thrust::lower_bound(vec.begin(), vec.end(), 3) - vec.begin(), 2);
    ASSERT_EQUAL(thrust::lower_bound(vec.begin(), vec.end(), 4) - vec.begin(), 2);
    ASSERT_EQUAL(thrust::lower_bound(vec.begin(), vec.end(), 5) - vec.begin(), 2);
    ASSERT_EQUAL(thrust::lower_bound(vec.begin(), vec.end(), 6) - vec.begin(), 3);
    ASSERT_EQUAL(thrust::lower_bound(vec.begin(), vec.end(), 7) - vec.begin(), 3);
    ASSERT_EQUAL(thrust::lower_bound(vec.begin(), vec.end(), 8) - vec.begin(), 4);
    ASSERT_EQUAL(thrust::lower_bound(vec.begin(), vec.end(), 9) - vec.begin(), 5);
}
DECLARE_VECTOR_UNITTEST(TestScalarLowerBoundSimple);


template<typename ForwardIterator, typename LessThanComparable>
ForwardIterator lower_bound(my_system, ForwardIterator first, ForwardIterator last, const LessThanComparable &value)
{
    *first = 13;
    return first;
}

void TestScalarLowerBoundDispatchExplicit()
{
    thrust::device_vector<int> vec(1);

    my_system sys;
    thrust::lower_bound(sys,
                        vec.begin(),
                        vec.end(),
                        0);

    ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestScalarLowerBoundDispatchExplicit);


void TestScalarLowerBoundDispatchImplicit()
{
    thrust::device_vector<int> vec(1);

    thrust::lower_bound(thrust::retag<my_system>(vec.begin()),
                        thrust::retag<my_system>(vec.end()),
                        0);

    ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestScalarLowerBoundDispatchImplicit);


template <class Vector>
void TestScalarUpperBoundSimple(void)
{
    typedef typename Vector::value_type T;

    Vector vec(5);

    vec[0] = 0;
    vec[1] = 2;
    vec[2] = 5;
    vec[3] = 7;
    vec[4] = 8;

    ASSERT_EQUAL(thrust::upper_bound(vec.begin(), vec.end(), 0) - vec.begin(), 1);
    ASSERT_EQUAL(thrust::upper_bound(vec.begin(), vec.end(), 1) - vec.begin(), 1);
    ASSERT_EQUAL(thrust::upper_bound(vec.begin(), vec.end(), 2) - vec.begin(), 2);
    ASSERT_EQUAL(thrust::upper_bound(vec.begin(), vec.end(), 3) - vec.begin(), 2);
    ASSERT_EQUAL(thrust::upper_bound(vec.begin(), vec.end(), 4) - vec.begin(), 2);
    ASSERT_EQUAL(thrust::upper_bound(vec.begin(), vec.end(), 5) - vec.begin(), 3);
    ASSERT_EQUAL(thrust::upper_bound(vec.begin(), vec.end(), 6) - vec.begin(), 3);
    ASSERT_EQUAL(thrust::upper_bound(vec.begin(), vec.end(), 7) - vec.begin(), 4);
    ASSERT_EQUAL(thrust::upper_bound(vec.begin(), vec.end(), 8) - vec.begin(), 5);
    ASSERT_EQUAL(thrust::upper_bound(vec.begin(), vec.end(), 9) - vec.begin(), 5);
}
DECLARE_VECTOR_UNITTEST(TestScalarUpperBoundSimple);


template<typename ForwardIterator, typename LessThanComparable>
ForwardIterator upper_bound(my_system, ForwardIterator first, ForwardIterator last, const LessThanComparable &value)
{
    *first = 13;
    return first;
}

void TestScalarUpperBoundDispatchExplicit()
{
    thrust::device_vector<int> vec(1);

    my_system sys;
    thrust::upper_bound(sys,
                        vec.begin(),
                        vec.end(),
                        0);

    ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestScalarUpperBoundDispatchExplicit);

void TestScalarUpperBoundDispatchImplicit()
{
    thrust::device_vector<int> vec(1);

    thrust::upper_bound(thrust::retag<my_system>(vec.begin()),
                        thrust::retag<my_system>(vec.end()),
                        0);

    ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestScalarUpperBoundDispatchImplicit);


template <class Vector>
void TestScalarBinarySearchSimple(void)
{
    typedef typename Vector::value_type T;

    Vector vec(5);

    vec[0] = 0;
    vec[1] = 2;
    vec[2] = 5;
    vec[3] = 7;
    vec[4] = 8;

    ASSERT_EQUAL(thrust::binary_search(vec.begin(), vec.end(), 0),  true);
    ASSERT_EQUAL(thrust::binary_search(vec.begin(), vec.end(), 1), false);
    ASSERT_EQUAL(thrust::binary_search(vec.begin(), vec.end(), 2),  true);
    ASSERT_EQUAL(thrust::binary_search(vec.begin(), vec.end(), 3), false);
    ASSERT_EQUAL(thrust::binary_search(vec.begin(), vec.end(), 4), false);
    ASSERT_EQUAL(thrust::binary_search(vec.begin(), vec.end(), 5),  true);
    ASSERT_EQUAL(thrust::binary_search(vec.begin(), vec.end(), 6), false);
    ASSERT_EQUAL(thrust::binary_search(vec.begin(), vec.end(), 7),  true);
    ASSERT_EQUAL(thrust::binary_search(vec.begin(), vec.end(), 8),  true);
    ASSERT_EQUAL(thrust::binary_search(vec.begin(), vec.end(), 9), false);
}
DECLARE_VECTOR_UNITTEST(TestScalarBinarySearchSimple);


template<typename ForwardIterator, typename LessThanComparable>
bool binary_search(my_system, ForwardIterator first, ForwardIterator last, const LessThanComparable &value)
{
    *first = 13;
    return false;
}

void TestScalarBinarySearchDispatchExplicit()
{
    thrust::device_vector<int> vec(1);

    my_system sys;
    thrust::binary_search(sys,
                          vec.begin(),
                          vec.end(),
                          0);

    ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestScalarBinarySearchDispatchExplicit);

void TestScalarBinarySearchDispatchImplicit()
{
    thrust::device_vector<int> vec(1);

    thrust::binary_search(thrust::retag<my_system>(vec.begin()),
                          thrust::retag<my_system>(vec.end()),
                          0);

    ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestScalarBinarySearchDispatchImplicit);


template <class Vector>
void TestScalarEqualRangeSimple(void)
{
    typedef typename Vector::value_type T;

    Vector vec(5);

    vec[0] = 0;
    vec[1] = 2;
    vec[2] = 5;
    vec[3] = 7;
    vec[4] = 8;

    ASSERT_EQUAL(thrust::equal_range(vec.begin(), vec.end(), 0).first - vec.begin(), 0);
    ASSERT_EQUAL(thrust::equal_range(vec.begin(), vec.end(), 1).first - vec.begin(), 1);
    ASSERT_EQUAL(thrust::equal_range(vec.begin(), vec.end(), 2).first - vec.begin(), 1);
    ASSERT_EQUAL(thrust::equal_range(vec.begin(), vec.end(), 3).first - vec.begin(), 2);
    ASSERT_EQUAL(thrust::equal_range(vec.begin(), vec.end(), 4).first - vec.begin(), 2);
    ASSERT_EQUAL(thrust::equal_range(vec.begin(), vec.end(), 5).first - vec.begin(), 2);
    ASSERT_EQUAL(thrust::equal_range(vec.begin(), vec.end(), 6).first - vec.begin(), 3);
    ASSERT_EQUAL(thrust::equal_range(vec.begin(), vec.end(), 7).first - vec.begin(), 3);
    ASSERT_EQUAL(thrust::equal_range(vec.begin(), vec.end(), 8).first - vec.begin(), 4);
    ASSERT_EQUAL(thrust::equal_range(vec.begin(), vec.end(), 9).first - vec.begin(), 5);
    
    ASSERT_EQUAL(thrust::equal_range(vec.begin(), vec.end(), 0).second - vec.begin(), 1);
    ASSERT_EQUAL(thrust::equal_range(vec.begin(), vec.end(), 1).second - vec.begin(), 1);
    ASSERT_EQUAL(thrust::equal_range(vec.begin(), vec.end(), 2).second - vec.begin(), 2);
    ASSERT_EQUAL(thrust::equal_range(vec.begin(), vec.end(), 3).second - vec.begin(), 2);
    ASSERT_EQUAL(thrust::equal_range(vec.begin(), vec.end(), 4).second - vec.begin(), 2);
    ASSERT_EQUAL(thrust::equal_range(vec.begin(), vec.end(), 5).second - vec.begin(), 3);
    ASSERT_EQUAL(thrust::equal_range(vec.begin(), vec.end(), 6).second - vec.begin(), 3);
    ASSERT_EQUAL(thrust::equal_range(vec.begin(), vec.end(), 7).second - vec.begin(), 4);
    ASSERT_EQUAL(thrust::equal_range(vec.begin(), vec.end(), 8).second - vec.begin(), 5);
    ASSERT_EQUAL(thrust::equal_range(vec.begin(), vec.end(), 9).second - vec.begin(), 5);
}
DECLARE_VECTOR_UNITTEST(TestScalarEqualRangeSimple);


template<typename ForwardIterator, typename LessThanComparable>
thrust::pair<ForwardIterator,ForwardIterator> equal_range(my_system, ForwardIterator first, ForwardIterator last, const LessThanComparable &value)
{
    *first = 13;
    return thrust::make_pair(first,first);
}

void TestScalarEqualRangeDispatchExplicit()
{
    thrust::device_vector<int> vec(1);

    my_system sys;
    thrust::equal_range(sys,
                        vec.begin(),
                        vec.end(),
                        0);

    ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestScalarEqualRangeDispatchExplicit);

void TestScalarEqualRangeDispatchImplicit()
{
    thrust::device_vector<int> vec(1);

    thrust::equal_range(thrust::retag<my_system>(vec.begin()),
                        thrust::retag<my_system>(vec.end()),
                        0);

    ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestScalarEqualRangeDispatchImplicit);

__THRUST_DISABLE_MSVC_POSSIBLE_LOSS_OF_DATA_WARNING_END
