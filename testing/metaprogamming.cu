#include <unittest/unittest.h>
#include <thrust/detail/mpl/math.h>

void TestLog2(void)
{
    unsigned int result;
    
    result = thrust::detail::mpl::math::log2<  1>::value;   ASSERT_EQUAL(result, 0);
    result = thrust::detail::mpl::math::log2<  2>::value;   ASSERT_EQUAL(result, 1);
    result = thrust::detail::mpl::math::log2<  3>::value;   ASSERT_EQUAL(result, 1);
    result = thrust::detail::mpl::math::log2<  4>::value;   ASSERT_EQUAL(result, 2);
    result = thrust::detail::mpl::math::log2<  5>::value;   ASSERT_EQUAL(result, 2);
    result = thrust::detail::mpl::math::log2<  6>::value;   ASSERT_EQUAL(result, 2);
    result = thrust::detail::mpl::math::log2<  7>::value;   ASSERT_EQUAL(result, 2);
    result = thrust::detail::mpl::math::log2<  8>::value;   ASSERT_EQUAL(result, 3);
    result = thrust::detail::mpl::math::log2<  9>::value;   ASSERT_EQUAL(result, 3);
    result = thrust::detail::mpl::math::log2< 15>::value;   ASSERT_EQUAL(result, 3);
    result = thrust::detail::mpl::math::log2< 16>::value;   ASSERT_EQUAL(result, 4);
    result = thrust::detail::mpl::math::log2< 17>::value;   ASSERT_EQUAL(result, 4);
    result = thrust::detail::mpl::math::log2<127>::value;   ASSERT_EQUAL(result, 6);
    result = thrust::detail::mpl::math::log2<128>::value;   ASSERT_EQUAL(result, 7);
    result = thrust::detail::mpl::math::log2<129>::value;   ASSERT_EQUAL(result, 7);
    result = thrust::detail::mpl::math::log2<256>::value;   ASSERT_EQUAL(result, 8);
    result = thrust::detail::mpl::math::log2<511>::value;   ASSERT_EQUAL(result, 8);
    result = thrust::detail::mpl::math::log2<512>::value;   ASSERT_EQUAL(result, 9);
}
DECLARE_UNITTEST(TestLog2);

