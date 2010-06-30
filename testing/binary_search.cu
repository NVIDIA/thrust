#include <unittest/unittest.h>
#include <thrust/binary_search.h>

#include <thrust/sequence.h>
#include <thrust/sort.h>

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

