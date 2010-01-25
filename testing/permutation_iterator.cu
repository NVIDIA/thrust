#include <thrusttest/unittest.h>
#include <thrust/iterator/permutation_iterator.h>

#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <thrust/sequence.h>

template <class Vector>
void TestPermutationIteratorSimple(void)
{
    typedef typename Vector::value_type T;
    typedef typename Vector::iterator   Iterator;

    Vector source(8);
    Vector indices(4);
    
    // initialize input
    thrust::sequence(source.begin(), source.end(), 1);

    indices[0] = 3;
    indices[1] = 0;
    indices[2] = 5;
    indices[3] = 7;
   
    thrust::permutation_iterator<Iterator, Iterator> begin(source.begin(), indices.begin());
    thrust::permutation_iterator<Iterator, Iterator> end(source.begin(),   indices.end());

    ASSERT_EQUAL(end - begin, 4);
    ASSERT_EQUAL((begin + 4) == end, true);

    ASSERT_EQUAL((T) *begin, 4);

    begin++;
    end--;

    ASSERT_EQUAL((T) *begin, 1);
    ASSERT_EQUAL((T) *end,   8);
    ASSERT_EQUAL(end - begin, 2);

    end--;

    *begin = 10;
    *end   = 20;

    ASSERT_EQUAL(source[0], 10);
    ASSERT_EQUAL(source[1],  2);
    ASSERT_EQUAL(source[2],  3);
    ASSERT_EQUAL(source[3],  4);
    ASSERT_EQUAL(source[4],  5);
    ASSERT_EQUAL(source[5], 20);
    ASSERT_EQUAL(source[6],  7);
    ASSERT_EQUAL(source[7],  8);
}
DECLARE_VECTOR_UNITTEST(TestPermutationIteratorSimple);

template <class Vector>
void TestPermutationIteratorGather(void)
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
   
    thrust::permutation_iterator<Iterator, Iterator> p_source(source.begin(), indices.begin());

    thrust::copy(p_source, p_source + 4, output.begin());

    ASSERT_EQUAL(output[0], 4);
    ASSERT_EQUAL(output[1], 1);
    ASSERT_EQUAL(output[2], 6);
    ASSERT_EQUAL(output[3], 8);
}
DECLARE_VECTOR_UNITTEST(TestPermutationIteratorGather);

template <class Vector>
void TestPermutationIteratorScatter(void)
{
    typedef typename Vector::iterator Iterator;

    Vector source(4, 10);
    Vector indices(4);
    Vector output(8);
    
    // initialize output
    thrust::sequence(output.begin(), output.end(), 1);

    indices[0] = 3;
    indices[1] = 0;
    indices[2] = 5;
    indices[3] = 7;
   
    // construct transform_iterator
    thrust::permutation_iterator<Iterator, Iterator> p_output(output.begin(), indices.begin());

    thrust::copy(source.begin(), source.end(), p_output);

    ASSERT_EQUAL(output[0], 10);
    ASSERT_EQUAL(output[1],  2);
    ASSERT_EQUAL(output[2],  3);
    ASSERT_EQUAL(output[3], 10);
    ASSERT_EQUAL(output[4],  5);
    ASSERT_EQUAL(output[5], 10);
    ASSERT_EQUAL(output[6],  7);
    ASSERT_EQUAL(output[7], 10);
}
DECLARE_VECTOR_UNITTEST(TestPermutationIteratorScatter);

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
    thrust::permutation_iterator<Iterator, Iterator> iter(source.begin(), indices.begin());

    thrust::copy(thrust::make_permutation_iterator(source.begin(), indices.begin()),
                 thrust::make_permutation_iterator(source.begin(), indices.begin()) + 4,
                 output.begin());

    ASSERT_EQUAL(output[0], 4);
    ASSERT_EQUAL(output[1], 1);
    ASSERT_EQUAL(output[2], 6);
    ASSERT_EQUAL(output[3], 8);
}
DECLARE_VECTOR_UNITTEST(TestMakePermutationIterator);

template <typename Vector>
void TestPermutationIteratorReduce(void)
{
    typedef typename Vector::value_type T;
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
    thrust::permutation_iterator<Iterator, Iterator> iter(source.begin(), indices.begin());

    T result1 = thrust::reduce(thrust::make_permutation_iterator(source.begin(), indices.begin()),
                               thrust::make_permutation_iterator(source.begin(), indices.begin()) + 4);

    ASSERT_EQUAL(result1, 19);
    
    T result2 = thrust::transform_reduce(thrust::make_permutation_iterator(source.begin(), indices.begin()),
                                         thrust::make_permutation_iterator(source.begin(), indices.begin()) + 4,
                                         thrust::negate<T>(),
                                         T(0),
                                         thrust::plus<T>());
    ASSERT_EQUAL(result2, -19);
};
DECLARE_VECTOR_UNITTEST(TestPermutationIteratorReduce);

