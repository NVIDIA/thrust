#include <thrusttest/unittest.h>
#include <thrust/detail/trivial_sequence.h>

#include <thrust/iterator/zip_iterator.h> 
#include <thrust/sort.h>
#include <thrust/copy.h>

template <typename Iterator>
void test(Iterator first, Iterator last)
{
    thrust::detail::trivial_sequence<Iterator> ts(first, last);

    thrust::sort(ts.begin(), ts.end());

    thrust::copy(ts.begin(), ts.end(), first);
}

template <class Vector>
void TestTrivialSequenceSort(void)
{
    Vector A(5);  A[0] =  0;  A[1] =  2;  A[2] =  1;  A[3] =  0;  A[4] =  1;  
    Vector B(5);  B[0] = 11;  B[1] = 11;  B[2] = 13;  B[3] = 10;  B[4] = 12;

    test(thrust::make_zip_iterator(thrust::make_tuple(A.begin(), B.begin())),
         thrust::make_zip_iterator(thrust::make_tuple(A.end(),   B.end())));

    ASSERT_EQUAL(A[0], 0);  ASSERT_EQUAL(B[0], 10); 
    ASSERT_EQUAL(A[1], 0);  ASSERT_EQUAL(B[1], 11); 
    ASSERT_EQUAL(A[2], 1);  ASSERT_EQUAL(B[2], 12); 
    ASSERT_EQUAL(A[3], 1);  ASSERT_EQUAL(B[3], 13); 
    ASSERT_EQUAL(A[4], 2);  ASSERT_EQUAL(B[4], 11); 
}
DECLARE_VECTOR_UNITTEST(TestTrivialSequenceSort);

