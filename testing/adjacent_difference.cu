#include <thrusttest/unittest.h>
#include <thrust/adjacent_difference.h>

template <class Vector>
void TestAjacentDifferenceSimple(void)
{
    typedef typename Vector::value_type T;

    Vector input(3);
    Vector output(3);
    input[0] = 1; input[1] = 4; input[2] = 6;

    typename Vector::iterator result;
    
    result = thrust::adjacent_difference(input.begin(), input.end(), output.begin());

    ASSERT_EQUAL(result - output.begin(), 3);
    ASSERT_EQUAL(output[0], 1);
    ASSERT_EQUAL(output[1], 3);
    ASSERT_EQUAL(output[2], 2);
    
    result = thrust::adjacent_difference(input.begin(), input.end(), output.begin(), thrust::plus<T>());
    
    ASSERT_EQUAL(result - output.begin(), 3);
    ASSERT_EQUAL(output[0],  1);
    ASSERT_EQUAL(output[1],  5);
    ASSERT_EQUAL(output[2], 10);
    
    // test in-place operation, result and first are permitted to be the same
    result = thrust::adjacent_difference(input.begin(), input.end(), input.begin());

    ASSERT_EQUAL(result - input.begin(), 3);
    ASSERT_EQUAL(input[0], 1);
    ASSERT_EQUAL(input[1], 3);
    ASSERT_EQUAL(input[2], 2);
}
DECLARE_VECTOR_UNITTEST(TestAjacentDifferenceSimple);


template <typename T>
void TestAjacentDifference(const size_t n)
{
    thrust::host_vector<T>   h_input = thrusttest::random_samples<T>(n);
    thrust::device_vector<T> d_input = h_input;

    thrust::host_vector<T>   h_output(n);
    thrust::device_vector<T> d_output(n);

    typename thrust::host_vector<T>::iterator   h_result;
    typename thrust::device_vector<T>::iterator d_result;

    h_result = thrust::adjacent_difference(h_input.begin(), h_input.end(), h_output.begin());
    d_result = thrust::adjacent_difference(d_input.begin(), d_input.end(), d_output.begin());

    ASSERT_EQUAL(h_result - h_output.begin(), n);
    ASSERT_EQUAL(d_result - d_output.begin(), n);
    ASSERT_EQUAL(h_output, d_output);
    
    h_result = thrust::adjacent_difference(h_input.begin(), h_input.end(), h_output.begin(), thrust::plus<T>());
    d_result = thrust::adjacent_difference(d_input.begin(), d_input.end(), d_output.begin(), thrust::plus<T>());

    ASSERT_EQUAL(h_result - h_output.begin(), n);
    ASSERT_EQUAL(d_result - d_output.begin(), n);
    ASSERT_EQUAL(h_output, d_output);
    
    // in-place operation
    h_result = thrust::adjacent_difference(h_input.begin(), h_input.end(), h_input.begin(), thrust::plus<T>());
    d_result = thrust::adjacent_difference(d_input.begin(), d_input.end(), d_input.begin(), thrust::plus<T>());

    ASSERT_EQUAL(h_result - h_input.begin(), n);
    ASSERT_EQUAL(d_result - d_input.begin(), n);
    ASSERT_EQUAL(h_input, h_output); //computed previously
    ASSERT_EQUAL(d_input, d_output); //computed previously
    
}
DECLARE_VARIABLE_UNITTEST(TestAjacentDifference);

