#include <unittest/unittest.h>
#include <thrust/mismatch.h>

template <class Vector>
void TestMismatchSimple(void)
{
    typedef typename Vector::value_type T;

    Vector a(4); Vector b(4);
    a[0] = 1; b[0] = 1;
    a[1] = 2; b[1] = 2;
    a[2] = 3; b[2] = 4;
    a[3] = 4; b[3] = 3;

    ASSERT_EQUAL(thrust::mismatch(a.begin(), a.end(), b.begin()).first  - a.begin(), 2);
    ASSERT_EQUAL(thrust::mismatch(a.begin(), a.end(), b.begin()).second - b.begin(), 2);

    b[2] = 3;
    
    ASSERT_EQUAL(thrust::mismatch(a.begin(), a.end(), b.begin()).first  - a.begin(), 3);
    ASSERT_EQUAL(thrust::mismatch(a.begin(), a.end(), b.begin()).second - b.begin(), 3);
    
    b[3] = 4;
    
    ASSERT_EQUAL(thrust::mismatch(a.begin(), a.end(), b.begin()).first  - a.begin(), 4);
    ASSERT_EQUAL(thrust::mismatch(a.begin(), a.end(), b.begin()).second - b.begin(), 4);
}
DECLARE_VECTOR_UNITTEST(TestMismatchSimple);

