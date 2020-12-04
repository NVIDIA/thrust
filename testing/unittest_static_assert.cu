#include <unittest/runtime_static_assert.h>
#include <unittest/unittest.h>
#include <thrust/generate.h>

template<typename T>
struct dependent_false
{
    enum { value = false };
};

template<typename T>
struct static_assertion
{
    __host__ __device__
    int operator()() const
    {
        THRUST_STATIC_ASSERT(dependent_false<T>::value);
        return 0;
    }
};

template<typename V>
void TestStaticAssertAssert()
{
    V test(10);
    ASSERT_STATIC_ASSERT(thrust::generate(test.begin(), test.end(), static_assertion<int>()));
}
DECLARE_VECTOR_UNITTEST(TestStaticAssertAssert);
