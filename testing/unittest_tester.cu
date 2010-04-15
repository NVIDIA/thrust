#include <unittest/unittest.h>

void TestAssertEqual(void)
{
    ASSERT_EQUAL(0, 0);
    ASSERT_EQUAL(1, 1);
    ASSERT_EQUAL(-15.0f, -15.0f);
}
DECLARE_UNITTEST(TestAssertEqual);

void TestAssertLEqual(void)
{
    ASSERT_LEQUAL(0, 1);
    ASSERT_LEQUAL(0, 0);
}
DECLARE_UNITTEST(TestAssertLEqual);

void TestAssertGEqual(void)
{
    ASSERT_GEQUAL(1, 0);
    ASSERT_GEQUAL(0, 0);
}
DECLARE_UNITTEST(TestAssertGEqual);

