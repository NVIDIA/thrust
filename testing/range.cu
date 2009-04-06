#include <komradetest/unittest.h>
#include <komrade/range.h>


template <class Vector>
void TestRangeSimple(void)
{
    typedef typename Vector::value_type T;
    
    Vector v(5);

    komrade::range(v.begin(), v.end());

    ASSERT_EQUAL(v[0], 0);
    ASSERT_EQUAL(v[1], 1);
    ASSERT_EQUAL(v[2], 2);
    ASSERT_EQUAL(v[3], 3);
    ASSERT_EQUAL(v[4], 4);

    komrade::range(v.begin(), v.end(), 10);

    ASSERT_EQUAL(v[0], 10);
    ASSERT_EQUAL(v[1], 11);
    ASSERT_EQUAL(v[2], 12);
    ASSERT_EQUAL(v[3], 13);
    ASSERT_EQUAL(v[4], 14);
    
    komrade::range(v.begin(), v.end(), 10, 2);

    ASSERT_EQUAL(v[0], 10);
    ASSERT_EQUAL(v[1], 12);
    ASSERT_EQUAL(v[2], 14);
    ASSERT_EQUAL(v[3], 16);
    ASSERT_EQUAL(v[4], 18);
}
DECLARE_VECTOR_UNITTEST(TestRangeSimple);


template <typename T>
void TestRange(size_t n)
{
    komrade::host_vector<T>   h_data(n);
    komrade::device_vector<T> d_data(n);

    komrade::range(h_data.begin(), h_data.end());
    komrade::range(d_data.begin(), d_data.end());

    ASSERT_EQUAL(h_data, d_data);

    komrade::range(h_data.begin(), h_data.end(), T(10));
    komrade::range(d_data.begin(), d_data.end(), T(10));

    ASSERT_EQUAL(h_data, d_data);

    komrade::range(h_data.begin(), h_data.end(), T(10), T(2));
    komrade::range(d_data.begin(), d_data.end(), T(10), T(2));

    ASSERT_EQUAL(h_data, d_data);
}
DECLARE_VARIABLE_UNITTEST(TestRange);
