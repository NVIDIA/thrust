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

struct my_system : thrust::device_system<my_system> {};

template<typename Iterator, typename Difference>
void advance(my_system, Iterator &i, Difference n)
{
  ;
}

void TestAdvanceDispatchExplicit()
{
    thrust::device_vector<int> vec(1);

    thrust::device_vector<int>::iterator i = vec.begin();

    my_system sys;
    thrust::advance(sys, i, 9001);

    ASSERT_EQUAL(0, i - vec.begin());
}
DECLARE_UNITTEST(TestAdvanceDispatchExplicit);

