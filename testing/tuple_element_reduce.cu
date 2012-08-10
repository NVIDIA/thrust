#include <unittest/unittest.h>
#include <thrust/tuple_reduce.h>

using namespace unittest;

void TestTupleElementReduce(void)
{
    int sum_a = thrust::tuple_sum(
        thrust::make_tuple(
            1, 2, 3, 4, 5),
        0);
    ASSERT_EQUAL(sum_a, 15);

    thrust::tuple<int, unsigned int, double, float> b =
        thrust::make_tuple(4, 2, 10, 5);

    int min_b = thrust::tuple_min(b, thrust::get<0>(b));
    ASSERT_EQUAL(min_b, 2);

    int max_b = thrust::tuple_max(b, thrust::get<0>(b));
    ASSERT_EQUAL(max_b, 10);
}

DECLARE_UNITTEST(TestTupleElementReduce);
