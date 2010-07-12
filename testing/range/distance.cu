#include <unittest/unittest.h>
#include <thrust/range/algorithm/distance.h>
#include <thrust/pair.h>

// TODO expand this with other iterator types (forward, bidirectional, etc.)

template <typename Vector>
void TestRangeDistance(void)
{
    typedef typename Vector::value_type T;
    typedef typename Vector::iterator Iterator;

    Vector v(100);

    ASSERT_EQUAL(100, thrust::experimental::range::distance(v));

    Iterator i = v.begin();

    i++;

    ASSERT_EQUAL(99, thrust::experimental::range::distance(thrust::make_pair(i, v.end())));

    i += 49;

    ASSERT_EQUAL(50, thrust::experimental::range::distance(thrust::make_pair(i, v.end())));
    
    ASSERT_EQUAL(0, thrust::experimental::range::distance(thrust::make_pair(i, i)));
}
DECLARE_VECTOR_UNITTEST(TestRangeDistance);

