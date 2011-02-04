#include <unittest/unittest.h>

template <class Vector>
void TestVectorCppZeroSize(void)
{
    Vector v;
    ASSERT_EQUAL(v.size(), 0);
    ASSERT_EQUAL((v.begin() == v.end()), true);
}
DECLARE_VECTOR_UNITTEST(TestVectorCppZeroSize);


