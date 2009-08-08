#include <thrusttest/unittest.h>
#include <thrust/iterator/constant_iterator.h>

using namespace thrust::experimental;

void TestConstantIteratorIncrement(void)
{
    constant_iterator<int> lhs(0,0);
    constant_iterator<int> rhs(0,0);

    ASSERT_EQUAL(0, lhs - rhs);

    lhs++;

    ASSERT_EQUAL(1, lhs - rhs);
    
    lhs++;
    lhs++;
    
    ASSERT_EQUAL(3, lhs - rhs);

    lhs += 5;
    
    ASSERT_EQUAL(8, lhs - rhs);

    lhs -= 10;
    
    ASSERT_EQUAL(-2, lhs - rhs);
}
DECLARE_UNITTEST(TestConstantIteratorIncrement);

void TestConstantIteratorComparison(void)
{
    constant_iterator<int> iter1(0);
    constant_iterator<int> iter2(0);

    ASSERT_EQUAL(0, iter1 - iter2);
    ASSERT_EQUAL(true, iter1 == iter2);

    iter1++;
    
    ASSERT_EQUAL(1, iter1 - iter2);
    ASSERT_EQUAL(false, iter1 == iter2);
   
    iter2++;

    ASSERT_EQUAL(0, iter1 - iter2);
    ASSERT_EQUAL(true, iter1 == iter2);
  
    iter1 += 100;
    iter2 += 100;

    ASSERT_EQUAL(0, iter1 - iter2);
    ASSERT_EQUAL(true, iter1 == iter2);
}
DECLARE_UNITTEST(TestConstantIteratorComparison);


void TestMakeConstantIterator(void)
{
    // test one argument version
    constant_iterator<int> iter0 = make_constant_iterator<int>(13);

    ASSERT_EQUAL(13, *iter0);

    // test two argument version
    constant_iterator<int,int> iter1 = make_constant_iterator<int,int>(13, 7);

    ASSERT_EQUAL(13, *iter1);
    ASSERT_EQUAL(7, iter1 - iter0);
}
DECLARE_UNITTEST(TestMakeConstantIterator);

template<typename Vector>
void TestConstantIteratorCopy(void)
{
  typedef typename Vector::value_type T;
  typedef constant_iterator<int> ConstIter;

  Vector result(4);

  ConstIter begin = make_constant_iterator<int>(7);
  ConstIter end   = begin + result.size();
  thrust::copy(begin, end, result.begin());

  ASSERT_EQUAL(7, result[0]);
  ASSERT_EQUAL(7, result[1]);
  ASSERT_EQUAL(7, result[2]);
  ASSERT_EQUAL(7, result[3]);
};
DECLARE_VECTOR_UNITTEST(TestConstantIteratorCopy)

