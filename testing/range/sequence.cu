#include <unittest/unittest.h>
#include <thrust/range/algorithm/sequence.h>


template <class Vector>
void TestRangeSequenceSimple(void)
{
    typedef typename Vector::value_type T;
    
    Vector v(5);

    using namespace thrust::experimental::range;

    sequence(v);

    ASSERT_EQUAL(v[0], 0);
    ASSERT_EQUAL(v[1], 1);
    ASSERT_EQUAL(v[2], 2);
    ASSERT_EQUAL(v[3], 3);
    ASSERT_EQUAL(v[4], 4);

    sequence(v, 10);

    ASSERT_EQUAL(v[0], 10);
    ASSERT_EQUAL(v[1], 11);
    ASSERT_EQUAL(v[2], 12);
    ASSERT_EQUAL(v[3], 13);
    ASSERT_EQUAL(v[4], 14);
    
    sequence(v, 10, 2);

    ASSERT_EQUAL(v[0], 10);
    ASSERT_EQUAL(v[1], 12);
    ASSERT_EQUAL(v[2], 14);
    ASSERT_EQUAL(v[3], 16);
    ASSERT_EQUAL(v[4], 18);
}
DECLARE_VECTOR_UNITTEST(TestRangeSequenceSimple);


template <typename T>
void TestRangeSequence(size_t n)
{
    using namespace thrust::experimental::range;

    thrust::host_vector<T>   h_data(n);
    thrust::device_vector<T> d_data(n);

    sequence(h_data);
    sequence(d_data);

    ASSERT_EQUAL(h_data, d_data);

    sequence(h_data, T(10));
    sequence(d_data, T(10));

    ASSERT_EQUAL(h_data, d_data);

    sequence(h_data, T(10), T(2));
    sequence(d_data, T(10), T(2));

    ASSERT_EQUAL(h_data, d_data);
}
DECLARE_VARIABLE_UNITTEST(TestRangeSequence);

