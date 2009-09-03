#include <thrusttest/unittest.h>
#include <thrust/iterator/transform_iterator.h>

#include <thrust/copy.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>
#include <thrust/iterator/counting_iterator.h>

template <class Vector>
void TestTransformIterator(void)
{
    typedef typename Vector::value_type T;

    typedef thrust::negate<T> UnaryFunction;
    typedef typename Vector::iterator Iterator;

    Vector input(4);
    Vector output(4);
    
    // initialize input
    thrust::sequence(input.begin(), input.end(), 1);
   
    // construct transform_iterator
    thrust::transform_iterator<UnaryFunction, Iterator> iter(input.begin(), UnaryFunction());

    thrust::copy(iter, iter + 4, output.begin());

    ASSERT_EQUAL(output[0], -1);
    ASSERT_EQUAL(output[1], -2);
    ASSERT_EQUAL(output[2], -3);
    ASSERT_EQUAL(output[3], -4);

}
DECLARE_VECTOR_UNITTEST(TestTransformIterator);

template <class Vector>
void TestMakeTransformIterator(void)
{
    typedef typename Vector::value_type T;

    typedef thrust::negate<T> UnaryFunction;
    typedef typename Vector::iterator Iterator;

    Vector input(4);
    Vector output(4);
    
    // initialize input
    thrust::sequence(input.begin(), input.end(), 1);
   
    // construct transform_iterator
    thrust::transform_iterator<UnaryFunction, Iterator> iter(input.begin(), UnaryFunction());

    thrust::copy(thrust::make_transform_iterator(input.begin(), UnaryFunction()), 
                 thrust::make_transform_iterator(input.end(), UnaryFunction()), 
                 output.begin());

    ASSERT_EQUAL(output[0], -1);
    ASSERT_EQUAL(output[1], -2);
    ASSERT_EQUAL(output[2], -3);
    ASSERT_EQUAL(output[3], -4);

}
DECLARE_VECTOR_UNITTEST(TestMakeTransformIterator);

