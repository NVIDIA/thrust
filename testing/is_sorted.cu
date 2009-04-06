#include <komradetest/unittest.h>
#include <komrade/is_sorted.h>
#include <komrade/sort.h>

template <class Vector>
void TestIsSortedSimple(void)
{
    typedef typename Vector::value_type T;

    Vector v(4);
    v[0] = 0; v[1] = 5; v[2] = 8; v[3] = 0;

    ASSERT_EQUAL(komrade::is_sorted(v.begin(), v.begin() + 0), true);
    ASSERT_EQUAL(komrade::is_sorted(v.begin(), v.begin() + 1), true);

    // the following line crashes gcc 4.3
#if (__GNUC__ == 4) && (__GNUC_MINOR__ == 3)
    // do nothing
#else
    // compile this line on other compilers
    ASSERT_EQUAL(komrade::is_sorted(v.begin(), v.begin() + 2), true);
#endif // GCC

    ASSERT_EQUAL(komrade::is_sorted(v.begin(), v.begin() + 3), true);
    ASSERT_EQUAL(komrade::is_sorted(v.begin(), v.begin() + 4), false);

    ASSERT_EQUAL(komrade::is_sorted(v.begin(), v.begin() + 3, komrade::less<T>()),    true);

    ASSERT_EQUAL(komrade::is_sorted(v.begin(), v.begin() + 1, komrade::greater<T>()), true);
    ASSERT_EQUAL(komrade::is_sorted(v.begin(), v.begin() + 4, komrade::greater<T>()), false);

    ASSERT_EQUAL(komrade::is_sorted(v.begin(), v.end()), false);
}
DECLARE_VECTOR_UNITTEST(TestIsSortedSimple);


template <class Vector>
void TestIsSorted(void)
{
    typedef typename Vector::value_type T;

    const size_t n = (1 << 16) + 13;

    Vector v = komradetest::random_integers<T>(n);

    v[0] = 1;
    v[1] = 0;

    ASSERT_EQUAL(komrade::is_sorted(v.begin(), v.end()), false);

    komrade::sort(v.begin(), v.end());

    ASSERT_EQUAL(komrade::is_sorted(v.begin(), v.end()), true);
}
DECLARE_VECTOR_UNITTEST(TestIsSorted);
