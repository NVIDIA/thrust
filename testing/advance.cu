#include <unittest/unittest.h>
#include <thrust/advance.h>
#include <thrust/sequence.h>

// TODO expand this with other iterator types (forward, bidirectional, etc.)

template <typename Vector>
void TestAdvance(void)
{
    typedef typename Vector::value_type T;
    typedef typename Vector::iterator Iterator;

    Vector v(100);
    thrust::sequence(v.begin(), v.end());

    Iterator i = v.begin();

    thrust::advance(i, 7);

    ASSERT_EQUAL(*i, T(7));
    
    thrust::advance(i, 13);

    ASSERT_EQUAL(*i, T(20));
    
    thrust::advance(i, -10);

    ASSERT_EQUAL(*i, T(10));
}
DECLARE_VECTOR_UNITTEST(TestAdvance);

template<typename Iterator, typename Difference>
void advance(my_system &system, Iterator &i, Difference n)
{
  system.validate_dispatch();
}

void TestAdvanceDispatchExplicit()
{
    thrust::device_vector<int> vec(1);

    thrust::device_vector<int>::iterator i = vec.begin();

    my_system sys(0);
    thrust::advance(sys, i, 9001);

    ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestAdvanceDispatchExplicit);

