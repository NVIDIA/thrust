#include <thrusttest/unittest.h>
#include <thrust/iterator/permutation_iterator.h>

#include <thrust/copy.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

template <class Vector>
void TestPermutationIterator(void)
{
    typedef typename Vector::iterator Iterator;

    Vector source(8);
    Vector indices(4);
    Vector output(4, 10);
    
    // initialize input
    thrust::sequence(source.begin(), source.end(), 1);

    indices[0] = 3;
    indices[1] = 0;
    indices[2] = 5;
    indices[3] = 7;
   
    // construct transform_iterator
    thrust::permutation_iterator<Iterator, Iterator> iter(input.begin(), indices.begin());

    thrust::copy(iter, iter + 4, output.begin());

    ASSERT_EQUAL(output[0], 4);
    ASSERT_EQUAL(output[1], 1);
    ASSERT_EQUAL(output[2], 6);
    ASSERT_EQUAL(output[3], 8);

}
DECLARE_VECTOR_UNITTEST(TestPermutationIterator);

template <class Vector>
void TestMakePermutationIterator(void)
{
    typedef typename Vector::iterator Iterator;

    Vector source(8);
    Vector indices(4);
    Vector output(4, 10);
    
    // initialize input
    thrust::sequence(source.begin(), source.end(), 1);

    indices[0] = 3;
    indices[1] = 0;
    indices[2] = 5;
    indices[3] = 7;
   
    // construct transform_iterator
    thrust::permutation_iterator<Iterator, Iterator> iter(input.begin(), indices.begin());

    thrust::copy(thrust::make_permutation_iterator(source.begin(), indices.begin()),
                 thrust::make_permutation_iterator(source.begin(), indices.begin()) + 4,
                 output.begin());

    ASSERT_EQUAL(output[0], 4);
    ASSERT_EQUAL(output[1], 1);
    ASSERT_EQUAL(output[2], 6);
    ASSERT_EQUAL(output[3], 8);
}
DECLARE_VECTOR_UNITTEST(TestMakePermutationIterator);

