#include <thrusttest/unittest.h>
#include <thrust/transform_scan.h>
#include <thrust/functional.h>

template <class Vector>
void TestTransformScanSimple(void)
{
    typedef typename Vector::value_type T;

    Vector input(5);
    Vector result(5);
    Vector output(5);

    input[0] = 1; input[1] = 3; input[2] = -2; input[3] = 4; input[4] = -5;

    Vector input_copy(input);

    // inclusive scan
    thrust::transform_inclusive_scan(input.begin(), input.end(), output.begin(), thrust::negate<T>(), thrust::plus<T>());
    result[0] = -1; result[1] = -4; result[2] = -2; result[3] = -6; result[4] = -1;
    ASSERT_EQUAL(input,  input_copy);
    ASSERT_EQUAL(output, result);
    
    // exclusive scan with 0 init
    thrust::transform_exclusive_scan(input.begin(), input.end(), output.begin(), thrust::negate<T>(), 0, thrust::plus<T>());
    result[0] = 0; result[1] = -1; result[2] = -4; result[3] = -2; result[4] = -6;
    ASSERT_EQUAL(input,  input_copy);
    ASSERT_EQUAL(output, result);
    
    // exclusive scan with nonzero init
    thrust::transform_exclusive_scan(input.begin(), input.end(), output.begin(), thrust::negate<T>(), 3, thrust::plus<T>());
    result[0] = 3; result[1] = 2; result[2] = -1; result[3] = 1; result[4] = -3;
    ASSERT_EQUAL(input,  input_copy);
    ASSERT_EQUAL(output, result);
    
    // inplace inclusive scan
    input = input_copy;
    thrust::transform_inclusive_scan(input.begin(), input.end(), input.begin(), thrust::negate<T>(), thrust::plus<T>());
    result[0] = -1; result[1] = -4; result[2] = -2; result[3] = -6; result[4] = -1;
    ASSERT_EQUAL(input, result);

    // inplace exclusive scan with init
    input = input_copy;
    thrust::transform_exclusive_scan(input.begin(), input.end(), input.begin(), thrust::negate<T>(), 3, thrust::plus<T>());
    result[0] = 3; result[1] = 2; result[2] = -1; result[3] = 1; result[4] = -3;
    ASSERT_EQUAL(input, result);
}
DECLARE_VECTOR_UNITTEST(TestTransformScanSimple);



template <typename T>
void TestTransformScan(const size_t n)
{
    thrust::host_vector<T>   h_input = thrusttest::random_integers<T>(n);
    for(size_t i = 0; i < n; i++)
        h_input[i] = (((int) h_input[i]) % 81 - 40) / 4.0;  // floats will be XX.0, XX.25, XX.5, or XX.75
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
DECLARE_VARIABLE_UNITTEST(TestTransformScan);

