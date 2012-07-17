#include <unittest/unittest.h>
#include <thrust/transform_scan.h>

#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/iterator_traits.h>

struct my_system : thrust::device_system<my_system> {};

template<typename InputIterator,
         typename OutputIterator,
         typename UnaryFunction,
         typename AssociativeOperator>
OutputIterator transform_inclusive_scan(my_system,
                                        InputIterator,
                                        InputIterator,
                                        OutputIterator result,
                                        UnaryFunction,
                                        AssociativeOperator)
{
    *result = 13;
    return result;
}

void TestTransformInclusiveScanDispatchExplicit()
{
    thrust::device_vector<int> vec(1);

    my_system sys;
    thrust::transform_inclusive_scan(sys,
                                     vec.begin(),
                                     vec.begin(),
                                     vec.begin(),
                                     0,
                                     0);

    ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestTransformInclusiveScanDispatchExplicit);

void TestTransformInclusiveScanDispatchImplicit()
{
    thrust::device_vector<int> vec(1);

    thrust::transform_inclusive_scan(thrust::retag<my_system>(vec.begin()),
                                     thrust::retag<my_system>(vec.begin()),
                                     thrust::retag<my_system>(vec.begin()),
                                     0,
                                     0);

    ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestTransformInclusiveScanDispatchImplicit);


template<typename InputIterator,
         typename OutputIterator,
         typename UnaryFunction,
         typename T,
         typename AssociativeOperator>
OutputIterator transform_exclusive_scan(my_system,
                                        InputIterator,
                                        InputIterator,
                                        OutputIterator result,
                                        UnaryFunction,
                                        T,
                                        AssociativeOperator)
{
    *result = 13;
    return result;
}

void TestTransformExclusiveScanDispatchExplicit()
{
    thrust::device_vector<int> vec(1);

    my_system sys;
    thrust::transform_exclusive_scan(sys,
                                     vec.begin(),
                                     vec.begin(),
                                     vec.begin(),
                                     0,
                                     0,
                                     0);

    ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestTransformExclusiveScanDispatchExplicit);

void TestTransformExclusiveScanDispatchImplicit()
{
    thrust::device_vector<int> vec(1);

    thrust::transform_exclusive_scan(thrust::retag<my_system>(vec.begin()),
                                     thrust::retag<my_system>(vec.begin()),
                                     thrust::retag<my_system>(vec.begin()),
                                     0,
                                     0,
                                     0);

    ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestTransformExclusiveScanDispatchImplicit);


template <class Vector>
void TestTransformScanSimple(void)
{
    typedef typename Vector::value_type T;

    typename Vector::iterator iter;

    Vector input(5);
    Vector result(5);
    Vector output(5);

    input[0] = 1; input[1] = 3; input[2] = -2; input[3] = 4; input[4] = -5;

    Vector input_copy(input);

    // inclusive scan
    iter = thrust::transform_inclusive_scan(input.begin(), input.end(), output.begin(), thrust::negate<T>(), thrust::plus<T>());
    result[0] = -1; result[1] = -4; result[2] = -2; result[3] = -6; result[4] = -1;
    ASSERT_EQUAL(iter - output.begin(), input.size());
    ASSERT_EQUAL(input,  input_copy);
    ASSERT_EQUAL(output, result);
    
    // exclusive scan with 0 init
    iter = thrust::transform_exclusive_scan(input.begin(), input.end(), output.begin(), thrust::negate<T>(), 0, thrust::plus<T>());
    result[0] = 0; result[1] = -1; result[2] = -4; result[3] = -2; result[4] = -6;
    ASSERT_EQUAL(iter - output.begin(), input.size());
    ASSERT_EQUAL(input,  input_copy);
    ASSERT_EQUAL(output, result);
    
    // exclusive scan with nonzero init
    iter = thrust::transform_exclusive_scan(input.begin(), input.end(), output.begin(), thrust::negate<T>(), 3, thrust::plus<T>());
    result[0] = 3; result[1] = 2; result[2] = -1; result[3] = 1; result[4] = -3;
    ASSERT_EQUAL(iter - output.begin(), input.size());
    ASSERT_EQUAL(input,  input_copy);
    ASSERT_EQUAL(output, result);
    
    // inplace inclusive scan
    input = input_copy;
    iter = thrust::transform_inclusive_scan(input.begin(), input.end(), input.begin(), thrust::negate<T>(), thrust::plus<T>());
    result[0] = -1; result[1] = -4; result[2] = -2; result[3] = -6; result[4] = -1;
    ASSERT_EQUAL(iter - input.begin(), input.size());
    ASSERT_EQUAL(input, result);

    // inplace exclusive scan with init
    input = input_copy;
    iter = thrust::transform_exclusive_scan(input.begin(), input.end(), input.begin(), thrust::negate<T>(), 3, thrust::plus<T>());
    result[0] = 3; result[1] = 2; result[2] = -1; result[3] = 1; result[4] = -3;
    ASSERT_EQUAL(iter - input.begin(), input.size());
    ASSERT_EQUAL(input, result);
}
DECLARE_VECTOR_UNITTEST(TestTransformScanSimple);


template <typename T>
struct TestTransformScan
{
    void operator()(const size_t n)
    {
        thrust::host_vector<T>   h_input = unittest::random_integers<T>(n);
        thrust::device_vector<T> d_input = h_input;

        thrust::host_vector<T>   h_output(n);
        thrust::device_vector<T> d_output(n);
        
        thrust::transform_inclusive_scan(h_input.begin(), h_input.end(), h_output.begin(), thrust::negate<T>(), thrust::plus<T>());
        thrust::transform_inclusive_scan(d_input.begin(), d_input.end(), d_output.begin(), thrust::negate<T>(), thrust::plus<T>());
        ASSERT_EQUAL(d_output, h_output);
        
        thrust::transform_exclusive_scan(h_input.begin(), h_input.end(), h_output.begin(), thrust::negate<T>(), (T) 11, thrust::plus<T>());
        thrust::transform_exclusive_scan(d_input.begin(), d_input.end(), d_output.begin(), thrust::negate<T>(), (T) 11, thrust::plus<T>());
        ASSERT_EQUAL(d_output, h_output);
        
        // in-place scans
        h_output = h_input;
        d_output = d_input;
        thrust::transform_inclusive_scan(h_output.begin(), h_output.end(), h_output.begin(), thrust::negate<T>(), thrust::plus<T>());
        thrust::transform_inclusive_scan(d_output.begin(), d_output.end(), d_output.begin(), thrust::negate<T>(), thrust::plus<T>());
        ASSERT_EQUAL(d_output, h_output);
        
        h_output = h_input;
        d_output = d_input;
        thrust::transform_exclusive_scan(h_output.begin(), h_output.end(), h_output.begin(), thrust::negate<T>(), (T) 11, thrust::plus<T>());
        thrust::transform_exclusive_scan(d_output.begin(), d_output.end(), d_output.begin(), thrust::negate<T>(), (T) 11, thrust::plus<T>());
        ASSERT_EQUAL(d_output, h_output);
    }
};
VariableUnitTest<TestTransformScan, IntegralTypes> TestTransformScanInstance;

template <class Vector>
void TestTransformScanCountingIterator(void)
{
    typedef typename Vector::value_type T;
    typedef typename thrust::iterator_system<typename Vector::iterator>::type space;

    thrust::counting_iterator<T, space> first(1);

    Vector result(3);

    thrust::transform_inclusive_scan(first, first + 3, result.begin(), thrust::negate<T>(), thrust::plus<T>());

    ASSERT_EQUAL(result[0], -1);
    ASSERT_EQUAL(result[1], -3);
    ASSERT_EQUAL(result[2], -6);
}
DECLARE_VECTOR_UNITTEST(TestTransformScanCountingIterator);

template <typename T>
struct TestTransformScanToDiscardIterator
{
    void operator()(const size_t n)
    {
        thrust::host_vector<T>   h_input = unittest::random_integers<T>(n);
        thrust::device_vector<T> d_input = h_input;

        thrust::discard_iterator<> reference(n);
        
        thrust::discard_iterator<> h_result =
          thrust::transform_inclusive_scan(h_input.begin(),
                                           h_input.end(),
                                           thrust::make_discard_iterator(),
                                           thrust::negate<T>(),
                                           thrust::plus<T>());

        thrust::discard_iterator<> d_result =
          thrust::transform_inclusive_scan(d_input.begin(),
                                           d_input.end(),
                                           thrust::make_discard_iterator(),
                                           thrust::negate<T>(),
                                           thrust::plus<T>());
        ASSERT_EQUAL_QUIET(reference, h_result);
        ASSERT_EQUAL_QUIET(reference, d_result);
        
        h_result =
          thrust::transform_exclusive_scan(h_input.begin(),
                                           h_input.end(),
                                           thrust::make_discard_iterator(),
                                           thrust::negate<T>(),
                                           (T) 11,
                                           thrust::plus<T>());

        d_result =
          thrust::transform_exclusive_scan(d_input.begin(),
                                           d_input.end(),
                                           thrust::make_discard_iterator(),
                                           thrust::negate<T>(),
                                           (T) 11,
                                           thrust::plus<T>());

        ASSERT_EQUAL_QUIET(reference, h_result);
        ASSERT_EQUAL_QUIET(reference, d_result);
    }
};
VariableUnitTest<TestTransformScanToDiscardIterator, IntegralTypes> TestTransformScanToDiscardIteratorInstance;

