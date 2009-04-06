#include <komradetest/unittest.h>
#include <komrade/equal.h>

#include <komrade/functional.h>

template <class Vector>
void TestEqualSimple(void)
{
    typedef typename Vector::value_type T;

    Vector v1(5);
    Vector v2(5);
    v1[0] = 5; v1[1] = 2; v1[2] = 0; v1[3] = 0; v1[4] = 0;
    v2[0] = 5; v2[1] = 2; v2[2] = 0; v2[3] = 6; v2[4] = 1;

    ASSERT_EQUAL(komrade::equal(v1.begin(), v1.end(), v1.begin()), true);
    ASSERT_EQUAL(komrade::equal(v1.begin(), v1.end(), v2.begin()), false);
    ASSERT_EQUAL(komrade::equal(v2.begin(), v2.end(), v2.begin()), true);
    
    ASSERT_EQUAL(komrade::equal(v1.begin(), v1.begin() + 0, v1.begin()), true);
    ASSERT_EQUAL(komrade::equal(v1.begin(), v1.begin() + 1, v1.begin()), true);
    ASSERT_EQUAL(komrade::equal(v1.begin(), v1.begin() + 3, v2.begin()), true);
    ASSERT_EQUAL(komrade::equal(v1.begin(), v1.begin() + 4, v2.begin()), false);
    
    ASSERT_EQUAL(komrade::equal(v1.begin(), v1.end(), v2.begin(), komrade::less_equal<T>()), true);
    ASSERT_EQUAL(komrade::equal(v1.begin(), v1.end(), v2.begin(), komrade::greater<T>()),    false);
}
DECLARE_VECTOR_UNITTEST(TestEqualSimple);

template <typename T>
void TestEqual(const size_t n)
{
    komrade::host_vector<T>   h_data1 = komradetest::random_samples<T>(n);
    komrade::host_vector<T>   h_data2 = komradetest::random_samples<T>(n);
    komrade::device_vector<T> d_data1 = h_data1;
    komrade::device_vector<T> d_data2 = h_data2;

    //empty ranges
    ASSERT_EQUAL(komrade::equal(h_data1.begin(), h_data1.begin(), h_data1.begin()), true);
    ASSERT_EQUAL(komrade::equal(d_data1.begin(), d_data1.begin(), d_data1.begin()), true);
    
    //symmetric cases
    ASSERT_EQUAL(komrade::equal(h_data1.begin(), h_data1.end(), h_data1.begin()), true);
    ASSERT_EQUAL(komrade::equal(d_data1.begin(), d_data1.end(), d_data1.begin()), true);

    if (n > 0)
    {
        h_data1[0] = 0; h_data2[0] = 1;
        d_data1[0] = 0; d_data2[0] = 1;

        //different vectors
        ASSERT_EQUAL(komrade::equal(h_data1.begin(), h_data1.end(), h_data2.begin()), false);
        ASSERT_EQUAL(komrade::equal(d_data1.begin(), d_data1.end(), d_data2.begin()), false);

        //different predicates
        ASSERT_EQUAL(komrade::equal(h_data1.begin(), h_data1.begin() + 1, h_data2.begin(), komrade::less<T>()), true);
        ASSERT_EQUAL(komrade::equal(d_data1.begin(), d_data1.begin() + 1, d_data2.begin(), komrade::less<T>()), true);
        ASSERT_EQUAL(komrade::equal(h_data1.begin(), h_data1.begin() + 1, h_data2.begin(), komrade::greater<T>()), false);
        ASSERT_EQUAL(komrade::equal(d_data1.begin(), d_data1.begin() + 1, d_data2.begin(), komrade::greater<T>()), false);
    }
}
DECLARE_VARIABLE_UNITTEST(TestEqual);

