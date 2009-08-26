#include <thrusttest/unittest.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sort.h>
#include <thrust/binary_search.h>

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

void TestCountingIteratorLowerBound(void)
{
    size_t n = 10000;
    const size_t M = 100;

    thrust::host_vector<unsigned int> h_data = thrusttest::random_integers<unsigned int>(n);
    for(unsigned int i = 0; i < n; ++i)
      h_data[i] %= M;

    thrust::sort(h_data.begin(), h_data.end());

    thrust::device_vector<unsigned int> d_data = h_data;

    thrust::experimental::counting_iterator<unsigned int> search_begin(0);
    thrust::experimental::counting_iterator<unsigned int> search_end(M);


    thrust::host_vector<unsigned int> h_result(M);
    thrust::device_vector<unsigned int> d_result(M);


    thrust::experimental::lower_bound(h_data.begin(), h_data.end(),
                                      search_begin, search_end,
                                      h_result.begin());

    thrust::experimental::lower_bound(d_data.begin(), d_data.end(),
                                      search_begin, search_end,
                                      d_result.begin());

    ASSERT_EQUAL(h_result, d_result);
}
DECLARE_UNITTEST(TestCountingIteratorLowerBound);

