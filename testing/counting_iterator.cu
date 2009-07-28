#include <thrusttest/unittest.h>
#include <thrust/iterator/counting_iterator.h>

void TestCountingIteratorIncrement(void)
{
    thrust::experimental::counting_iterator<int> iter(0);

    ASSERT_EQUAL(*iter, 0);

    iter++;

    ASSERT_EQUAL(*iter, 1);
    
    iter++;
    iter++;
    
    ASSERT_EQUAL(*iter, 3);

    iter += 5;
    
    ASSERT_EQUAL(*iter, 8);

    iter -= 10;
    
    ASSERT_EQUAL(*iter, -2);
}
DECLARE_UNITTEST(TestCountingIteratorIncrement);

void TestCountingIteratorComparison(void)
{
    thrust::experimental::counting_iterator<int> iter1(0);
    thrust::experimental::counting_iterator<int> iter2(0);

    ASSERT_EQUAL(iter1 - iter2, 0);
    ASSERT_EQUAL(iter1 == iter2, true);

    iter1++;
    
    ASSERT_EQUAL(iter1 - iter2, 1);
    ASSERT_EQUAL(iter1 == iter2, false);
   
    iter2++;

    ASSERT_EQUAL(iter1 - iter2, 0);
    ASSERT_EQUAL(iter1 == iter2, true);
  
    iter1 += 100;
    iter2 += 100;

    ASSERT_EQUAL(iter1 - iter2, 0);
    ASSERT_EQUAL(iter1 == iter2, true);
}
DECLARE_UNITTEST(TestCountingIteratorComparison);


