#include <thrusttest/unittest.h>
#include <thrust/swap_ranges.h>

template <class Vector>
void TestSwapRangesSimple(void)
{
    typedef typename Vector::value_type T;

    Vector v1(5);
    v1[0] = 0; v1[1] = 1; v1[2] = 2; v1[3] = 3; v1[4] = 4;

    Vector v2(5);
    v2[0] = 5; v2[1] = 6; v2[2] = 7; v2[3] = 8; v2[4] = 9;

    thrust::swap_ranges(v1.begin(), v1.end(), v2.begin());

    ASSERT_EQUAL(v1[0], 5);
    ASSERT_EQUAL(v1[1], 6);
    ASSERT_EQUAL(v1[2], 7);
    ASSERT_EQUAL(v1[3], 8);
    ASSERT_EQUAL(v1[4], 9);
    
    ASSERT_EQUAL(v2[0], 0);
    ASSERT_EQUAL(v2[1], 1);
    ASSERT_EQUAL(v2[2], 2);
    ASSERT_EQUAL(v2[3], 3);
    ASSERT_EQUAL(v2[4], 4);
}
DECLARE_VECTOR_UNITTEST(TestSwapRangesSimple);

template <class Vector>
void TestSwapMixedRanges(void)
{
    typedef typename Vector::value_type T;

    Vector v(5);
    v[0] = 0; v[1] = 1; v[2] = 2; v[3] = 3; v[4] = 4;

    thrust::device_vector<T> h(5);
    h[0] = 5; h[1] = 6; h[2] = 7; h[3] = 8; h[4] = 9;
    
    thrust::device_vector<T> d(5);
    d[0] = 10; d[1] = 11; d[2] = 12; d[3] = 13; d[4] = 14;

    thrust::swap_ranges(v.begin(), v.end(), h.begin());

    ASSERT_EQUAL(v[0], 5);
    ASSERT_EQUAL(v[1], 6);
    ASSERT_EQUAL(v[2], 7);
    ASSERT_EQUAL(v[3], 8);
    ASSERT_EQUAL(v[4], 9);
    
    ASSERT_EQUAL(h[0], 0);
    ASSERT_EQUAL(h[1], 1);
    ASSERT_EQUAL(h[2], 2);
    ASSERT_EQUAL(h[3], 3);
    ASSERT_EQUAL(h[4], 4);

    thrust::swap_ranges(v.begin(), v.end(), d.begin());
    
    ASSERT_EQUAL(d[0], 5);
    ASSERT_EQUAL(d[1], 6);
    ASSERT_EQUAL(d[2], 7);
    ASSERT_EQUAL(d[3], 8);
    ASSERT_EQUAL(d[4], 9);
    
    ASSERT_EQUAL(v[0], 10);
    ASSERT_EQUAL(v[1], 11);
    ASSERT_EQUAL(v[2], 12);
    ASSERT_EQUAL(v[3], 13);
    ASSERT_EQUAL(v[4], 14);
}
DECLARE_VECTOR_UNITTEST(TestSwapMixedRanges);


template <typename T>
void TestSwapRanges(const size_t n)
{
    thrust::host_vector<T> a1 = thrusttest::random_integers<T>(n);
    thrust::host_vector<T> a2 = thrusttest::random_integers<T>(n);

    thrust::host_vector<T>    h1 = a1;
    thrust::host_vector<T>    h2 = a2;
    thrust::device_vector<T>  d1 = a1;
    thrust::device_vector<T>  d2 = a2;
  
    thrust::swap_ranges(h1.begin(), h1.end(), h2.begin());
    thrust::swap_ranges(d1.begin(), d1.end(), d2.begin());

    ASSERT_EQUAL(h1, a2);  
    ASSERT_EQUAL(d1, a2);
    ASSERT_EQUAL(h2, a1);
    ASSERT_EQUAL(d2, a1);

    thrust::swap_ranges(h1.begin(), h1.end(), d2.begin());
    thrust::swap_ranges(d1.begin(), d1.end(), h2.begin());
    
    ASSERT_EQUAL(h1, a1);  
    ASSERT_EQUAL(d1, a1);
    ASSERT_EQUAL(h2, a2);
    ASSERT_EQUAL(h2, a2);
}
DECLARE_VARIABLE_UNITTEST(TestSwapRanges);
