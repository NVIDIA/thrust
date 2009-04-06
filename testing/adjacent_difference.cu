#include <komradetest/unittest.h>
#include <komrade/adjacent_difference.h>

template <class Vector>
void TestAjacentDifferenceSimple(void)
{
    typedef typename Vector::value_type T;

    Vector input(3);
    Vector output(3);
    input[0] = 1; input[1] = 4; input[2] = 6;

    typename Vector::iterator result;
    
    result = komrade::adjacent_difference(input.begin(), input.end(), output.begin());

    ASSERT_EQUAL(result - output.begin(), 3);
    ASSERT_EQUAL(output[0], 1);
    ASSERT_EQUAL(output[1], 3);
    ASSERT_EQUAL(output[2], 2);
    
    result = komrade::adjacent_difference(input.begin(), input.end(), output.begin(), komrade::plus<T>());
    
    ASSERT_EQUAL(result - output.begin(), 3);
    ASSERT_EQUAL(output[0],  1);
    ASSERT_EQUAL(output[1],  5);
    ASSERT_EQUAL(output[2], 10);
    
    // test in-place operation, result and first are permitted to be the same
    result = komrade::adjacent_difference(input.begin(), input.end(), input.begin());

    ASSERT_EQUAL(result - input.begin(), 3);
    ASSERT_EQUAL(input[0], 1);
    ASSERT_EQUAL(input[1], 3);
    ASSERT_EQUAL(input[2], 2);
}
DECLARE_VECTOR_UNITTEST(TestAjacentDifferenceSimple);


template <typename T>
void TestAjacentDifference(const size_t n)
{
    komrade::host_vector<T>   h_input = komradetest::random_samples<T>(n);
    komrade::device_vector<T> d_input = h_input;

    komrade::host_vector<T>   h_output(n);
    komrade::device_vector<T> d_output(n);

    typename komrade::host_vector<T>::iterator   h_result;
    typename komrade::device_vector<T>::iterator d_result;

    h_result = komrade::adjacent_difference(h_input.begin(), h_input.end(), h_output.begin());
    d_result = komrade::adjacent_difference(d_input.begin(), d_input.end(), d_output.begin());

    ASSERT_EQUAL(h_result - h_output.begin(), n);
    ASSERT_EQUAL(d_result - d_output.begin(), n);
    ASSERT_EQUAL(h_output, d_output);
    
    h_result = komrade::adjacent_difference(h_input.begin(), h_input.end(), h_output.begin(), komrade::plus<T>());
    d_result = komrade::adjacent_difference(d_input.begin(), d_input.end(), d_output.begin(), komrade::plus<T>());

    ASSERT_EQUAL(h_result - h_output.begin(), n);
    ASSERT_EQUAL(d_result - d_output.begin(), n);
    ASSERT_EQUAL(h_output, d_output);
    
    // in-place operation
    h_result = komrade::adjacent_difference(h_input.begin(), h_input.end(), h_input.begin(), komrade::plus<T>());
    d_result = komrade::adjacent_difference(d_input.begin(), d_input.end(), d_input.begin(), komrade::plus<T>());

    ASSERT_EQUAL(h_result - h_input.begin(), n);
    ASSERT_EQUAL(d_result - d_input.begin(), n);
    ASSERT_EQUAL(h_input, h_output); //computed previously
    ASSERT_EQUAL(d_input, d_output); //computed previously
    
}
DECLARE_VARIABLE_UNITTEST(TestAjacentDifference);

