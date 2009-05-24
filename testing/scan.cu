#include <thrusttest/unittest.h>
#include <thrust/scan.h>
#include <thrust/functional.h>

template <class Vector>
void TestScanSimple(void)
{
    typedef typename Vector::value_type T;

    Vector input(5);
    Vector result(5);
    Vector output(5);

    input[0] = 1; input[1] = 3; input[2] = -2; input[3] = 4; input[4] = -5;

    Vector input_copy(input);

    // inclusive scan
    thrust::inclusive_scan(input.begin(), input.end(), output.begin());
    result[0] = 1; result[1] = 4; result[2] = 2; result[3] = 6; result[4] = 1;
    ASSERT_EQUAL(input,  input_copy);
    ASSERT_EQUAL(output, result);
    
    // exclusive scan
    thrust::exclusive_scan(input.begin(), input.end(), output.begin(), 0);
    result[0] = 0; result[1] = 1; result[2] = 4; result[3] = 2; result[4] = 6;
    ASSERT_EQUAL(input,  input_copy);
    ASSERT_EQUAL(output, result);
    
    // exclusive scan with init
    thrust::exclusive_scan(input.begin(), input.end(), output.begin(), 3);
    result[0] = 3; result[1] = 4; result[2] = 7; result[3] = 5; result[4] = 9;
    ASSERT_EQUAL(input,  input_copy);
    ASSERT_EQUAL(output, result);
    
    // inclusive scan with op
    thrust::inclusive_scan(input.begin(), input.end(), output.begin(), thrust::plus<T>());
    result[0] = 1; result[1] = 4; result[2] = 2; result[3] = 6; result[4] = 1;
    ASSERT_EQUAL(input,  input_copy);
    ASSERT_EQUAL(output, result);

    // exclusive scan with init and op
    thrust::exclusive_scan(input.begin(), input.end(), output.begin(), 3, thrust::plus<T>());
    result[0] = 3; result[1] = 4; result[2] = 7; result[3] = 5; result[4] = 9;
    ASSERT_EQUAL(input,  input_copy);
    ASSERT_EQUAL(output, result);

    // inplace inclusive scan
    input = input_copy;
    thrust::inclusive_scan(input.begin(), input.end(), input.begin());
    result[0] = 1; result[1] = 4; result[2] = 2; result[3] = 6; result[4] = 1;
    ASSERT_EQUAL(input, result);

    // inplace exclusive scan with init
    input = input_copy;
    thrust::exclusive_scan(input.begin(), input.end(), input.begin(), 3);
    result[0] = 3; result[1] = 4; result[2] = 7; result[3] = 5; result[4] = 9;
    ASSERT_EQUAL(input, result);

    // inplace exclusive scan with implicit init=0
    input = input_copy;
    thrust::exclusive_scan(input.begin(), input.end(), input.begin());
    result[0] = 0; result[1] = 1; result[2] = 4; result[3] = 2; result[4] = 6;
    ASSERT_EQUAL(input, result);
}
DECLARE_VECTOR_UNITTEST(TestScanSimple);


void TestInclusiveScan32(void)
{
    typedef int T;
    size_t n = 32;

    thrust::host_vector<T>   h_input = thrusttest::random_integers<T>(n);
    thrust::device_vector<T> d_input = h_input;
    
    thrust::host_vector<T>   h_output(n);
    thrust::device_vector<T> d_output(n);

    thrust::inclusive_scan(h_input.begin(), h_input.end(), h_output.begin());
    thrust::inclusive_scan(d_input.begin(), d_input.end(), d_output.begin());

    ASSERT_EQUAL(d_output, h_output);
}
DECLARE_UNITTEST(TestInclusiveScan32);


void TestExclusiveScan32(void)
{
    typedef int T;
    size_t n = 32;
    T init = 13;

    thrust::host_vector<T>   h_input = thrusttest::random_integers<T>(n);
    thrust::device_vector<T> d_input = h_input;
    
    thrust::host_vector<T>   h_output(n);
    thrust::device_vector<T> d_output(n);

    thrust::exclusive_scan(h_input.begin(), h_input.end(), h_output.begin(), init);
    thrust::exclusive_scan(d_input.begin(), d_input.end(), d_output.begin(), init);

    ASSERT_EQUAL(d_output, h_output);
}
DECLARE_UNITTEST(TestExclusiveScan32);


template <class IntVector, class FloatVector>
void TestScanMixedTypes(void)
{
    // make sure we get types for default args and operators correct
    IntVector int_input(4);
    int_input[0] = 1;
    int_input[1] = 2;
    int_input[2] = 3;
    int_input[3] = 4;

    FloatVector float_input(4);
    float_input[0] = 1.5;
    float_input[1] = 2.5;
    float_input[2] = 3.5;
    float_input[3] = 4.5;

    IntVector   int_output(4);
    FloatVector float_output(4);
     
    // float -> int should use using plus<int> operator by default
    thrust::inclusive_scan(float_input.begin(), float_input.end(), int_output.begin());
    ASSERT_EQUAL(int_output[0],  1);
    ASSERT_EQUAL(int_output[1],  3);
    ASSERT_EQUAL(int_output[2],  6);
    ASSERT_EQUAL(int_output[3], 10);
    
    // float -> float with plus<int> operator
    thrust::inclusive_scan(float_input.begin(), float_input.end(), float_output.begin(), thrust::plus<int>());
    ASSERT_EQUAL(float_output[0],  1.5);
    ASSERT_EQUAL(float_output[1],  3.0);
    ASSERT_EQUAL(float_output[2],  6.0);
    ASSERT_EQUAL(float_output[3], 10.0);
    
    // float -> int should use using plus<int> operator by default
    thrust::exclusive_scan(float_input.begin(), float_input.end(), int_output.begin());
    ASSERT_EQUAL(int_output[0], 0);
    ASSERT_EQUAL(int_output[1], 1);
    ASSERT_EQUAL(int_output[2], 3);
    ASSERT_EQUAL(int_output[3], 6);
    
    // float -> int should use using plus<int> operator by default
    thrust::exclusive_scan(float_input.begin(), float_input.end(), int_output.begin(), (float) 5.5);
    ASSERT_EQUAL(int_output[0],  5);
    ASSERT_EQUAL(int_output[1],  6);
    ASSERT_EQUAL(int_output[2],  8);
    ASSERT_EQUAL(int_output[3], 11);
    
    // int -> float should use using plus<float> operator by default
    thrust::inclusive_scan(int_input.begin(), int_input.end(), float_output.begin());
    ASSERT_EQUAL(float_output[0],  1.0);
    ASSERT_EQUAL(float_output[1],  3.0);
    ASSERT_EQUAL(float_output[2],  6.0);
    ASSERT_EQUAL(float_output[3], 10.0);
    
    // int -> float should use using plus<float> operator by default
    thrust::exclusive_scan(int_input.begin(), int_input.end(), float_output.begin(), (float) 5.5);
    ASSERT_EQUAL(float_output[0],  5.5);
    ASSERT_EQUAL(float_output[1],  6.5);
    ASSERT_EQUAL(float_output[2],  8.5);
    ASSERT_EQUAL(float_output[3], 11.5);
}
void TestScanMixedTypesHost(void)
{
    TestScanMixedTypes< thrust::host_vector<int>, thrust::host_vector<float> >();
}
DECLARE_UNITTEST(TestScanMixedTypesHost);
void TestScanMixedTypesDevice(void)
{
    TestScanMixedTypes< thrust::device_vector<int>, thrust::device_vector<float> >();
}
DECLARE_UNITTEST(TestScanMixedTypesDevice);


template <typename T>
void TestScan(const size_t n)
{
    thrust::host_vector<T>   h_input = thrusttest::random_integers<T>(n);
    for(size_t i = 0; i < n; i++)
        h_input[i] = (((int) h_input[i]) % 81 - 40) / 4.0;  // floats will be XX.0, XX.25, XX.5, or XX.75
    thrust::device_vector<T> d_input = h_input;

    thrust::host_vector<T>   h_output(n);
    thrust::device_vector<T> d_output(n);
    
    thrust::host_vector<float>   h_float_output(n);
    thrust::device_vector<float> d_float_output(n);
    thrust::host_vector<int>   h_int_output(n);
    thrust::device_vector<int> d_int_output(n);

    
    thrust::inclusive_scan(h_input.begin(), h_input.end(), h_output.begin());
    thrust::inclusive_scan(d_input.begin(), d_input.end(), d_output.begin());
    ASSERT_EQUAL(d_output, h_output);
    
    thrust::exclusive_scan(h_input.begin(), h_input.end(), h_output.begin());
    thrust::exclusive_scan(d_input.begin(), d_input.end(), d_output.begin());
    ASSERT_EQUAL(d_output, h_output);
    
    thrust::exclusive_scan(h_input.begin(), h_input.end(), h_output.begin(), (T) 11);
    thrust::exclusive_scan(d_input.begin(), d_input.end(), d_output.begin(), (T) 11);
    ASSERT_EQUAL(d_output, h_output);
    
    //mixed input/output types
    thrust::inclusive_scan(h_input.begin(), h_input.end(), h_float_output.begin());
    thrust::inclusive_scan(d_input.begin(), d_input.end(), d_float_output.begin());
    ASSERT_EQUAL(d_output, h_output);
    
    thrust::exclusive_scan(h_input.begin(), h_input.end(), h_float_output.begin(), (float) 3.1415);
    thrust::exclusive_scan(d_input.begin(), d_input.end(), d_float_output.begin(), (float) 3.1415);
    ASSERT_EQUAL(d_output, h_output);
    
    thrust::exclusive_scan(h_input.begin(), h_input.end(), h_float_output.begin(), (int) 3);
    thrust::exclusive_scan(d_input.begin(), d_input.end(), d_float_output.begin(), (int) 3);
    ASSERT_EQUAL(d_output, h_output);
    
    thrust::exclusive_scan(h_input.begin(), h_input.end(), h_int_output.begin(), (int) 3);
    thrust::exclusive_scan(d_input.begin(), d_input.end(), d_int_output.begin(), (int) 3);
    ASSERT_EQUAL(d_output, h_output);
    
    thrust::exclusive_scan(h_input.begin(), h_input.end(), h_int_output.begin(), (float) 3.1415);
    thrust::exclusive_scan(d_input.begin(), d_input.end(), d_int_output.begin(), (float) 3.1415);
    ASSERT_EQUAL(d_output, h_output);
    
    // in-place scan
    thrust::inclusive_scan(h_input.begin(), h_input.end(), h_input.begin());
    thrust::inclusive_scan(d_input.begin(), d_input.end(), d_input.begin());
    ASSERT_EQUAL(d_input, h_input);
}
DECLARE_VARIABLE_UNITTEST(TestScan);


template <typename T>
struct TestInclusiveScanPair
{
  void operator()(const size_t n)
  {
    thrusttest::random_integer<T> rnd;

    thrust::host_vector< thrusttest::test_pair<T,T> > h_input(n);
    for(size_t i = 0; i < n; i++){
        h_input[i].first  = rnd();
        h_input[i].second = rnd();
    }
    thrust::device_vector< thrusttest::test_pair<T,T> > d_input = h_input;
    
    thrust::host_vector< thrusttest::test_pair<T,T> > h_output(n);
    thrust::device_vector< thrusttest::test_pair<T,T> > d_output(n);

    //thrusttest::test_pair<T,T> init(13, 29);

    thrust::inclusive_scan(h_input.begin(), h_input.end(), h_output.begin());
    thrust::inclusive_scan(d_input.begin(), d_input.end(), d_output.begin());

    ASSERT_EQUAL(d_input, h_input);
    ASSERT_EQUAL(d_output, h_output);
  }
};
VariableUnitTest<TestInclusiveScanPair, IntegralTypes> TestInclusiveScanPairInstance;

