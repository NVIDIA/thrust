#include <unittest/unittest.h>

template <class Vector>
void TestVectorCppZeroSize(void)
{
    Vector v;
    ASSERT_EQUAL(v.size(), 0lu);
    ASSERT_EQUAL((v.begin() == v.end()), true);
}
DECLARE_INTEGRAL_VECTOR_UNITTEST(TestVectorCppZeroSize);

// NOTE: the above requires INTEGRAL because custom_numeric is not trivially destructible
// and the code path through destroy_range fails when compiling as C++ and not CUDA C++,
// because the cub backend is not found

