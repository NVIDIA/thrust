#include <komradetest/unittest.h>
#include <komrade/scan.h>
#include <komrade/functional.h>

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
    komrade::inclusive_scan(input.begin(), input.end(), output.begin());
    result[0] = 1; result[1] = 4; result[2] = 2; result[3] = 6; result[4] = 1;
    ASSERT_EQUAL(input,  input_copy);
    ASSERT_EQUAL(output, result);
    
    // exclusive scan
    komrade::exclusive_scan(input.begin(), input.end(), output.begin(), 0);
    result[0] = 0; result[1] = 1; result[2] = 4; result[3] = 2; result[4] = 6;
    ASSERT_EQUAL(input,  input_copy);
    ASSERT_EQUAL(output, result);
    
    // exclusive scan with init
    komrade::exclusive_scan(input.begin(), input.end(), output.begin(), 3);
    result[0] = 3; result[1] = 4; result[2] = 7; result[3] = 5; result[4] = 9;
    ASSERT_EQUAL(input,  input_copy);
    ASSERT_EQUAL(output, result);
    
    // inclusive scan with op
    komrade::inclusive_scan(input.begin(), input.end(), output.begin(), komrade::plus<T>());
    result[0] = 1; result[1] = 4; result[2] = 2; result[3] = 6; result[4] = 1;
    ASSERT_EQUAL(input,  input_copy);
    ASSERT_EQUAL(output, result);

    // exclusive scan with init and op
    komrade::exclusive_scan(input.begin(), input.end(), output.begin(), 3, komrade::plus<T>());
    result[0] = 3; result[1] = 4; result[2] = 7; result[3] = 5; result[4] = 9;
    ASSERT_EQUAL(input,  input_copy);
    ASSERT_EQUAL(output, result);

    // inplace inclusive scan
    input = input_copy;
    komrade::inclusive_scan(input.begin(), input.end(), input.begin());
    result[0] = 1; result[1] = 4; result[2] = 2; result[3] = 6; result[4] = 1;
    ASSERT_EQUAL(input, result);

    // inplace exclusive scan with init
    input = input_copy;
    komrade::exclusive_scan(input.begin(), input.end(), input.begin(), 3);
    result[0] = 3; result[1] = 4; result[2] = 7; result[3] = 5; result[4] = 9;
    ASSERT_EQUAL(input, result);

    // inplace exclusive scan with implicit init=0
    input = input_copy;
    komrade::exclusive_scan(input.begin(), input.end(), input.begin());
    result[0] = 0; result[1] = 1; result[2] = 4; result[3] = 2; result[4] = 6;
    ASSERT_EQUAL(input, result);
}
DECLARE_VECTOR_UNITTEST(TestScanSimple);


void TestInclusiveScan32(void)
{
    typedef int T;
    size_t n = 32;

    komrade::host_vector<T>   h_input = komradetest::random_integers<T>(n);
    komrade::device_vector<T> d_input = h_input;
    
    komrade::host_vector<T>   h_output(n);
    komrade::device_vector<T> d_output(n);

    komrade::inclusive_scan(h_input.begin(), h_input.end(), h_output.begin());
    komrade::inclusive_scan(d_input.begin(), d_input.end(), d_output.begin());

    ASSERT_EQUAL(d_output, h_output);
}
DECLARE_UNITTEST(TestInclusiveScan32);


void TestExclusiveScan32(void)
{
    typedef int T;
    size_t n = 32;
    T init = 13;

    komrade::host_vector<T>   h_input = komradetest::random_integers<T>(n);
    komrade::device_vector<T> d_input = h_input;
    
    komrade::host_vector<T>   h_output(n);
    komrade::device_vector<T> d_output(n);

    komrade::exclusive_scan(h_input.begin(), h_input.end(), h_output.begin(), init);
    komrade::exclusive_scan(d_input.begin(), d_input.end(), d_output.begin(), init);

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
    komrade::inclusive_scan(float_input.begin(), float_input.end(), int_output.begin());
    ASSERT_EQUAL(int_output[0],  1);
    ASSERT_EQUAL(int_output[1],  3);
    ASSERT_EQUAL(int_output[2],  6);
    ASSERT_EQUAL(int_output[3], 10);
    
    // float -> float with plus<int> operator
    komrade::inclusive_scan(float_input.begin(), float_input.end(), float_output.begin(), komrade::plus<int>());
    ASSERT_EQUAL(float_output[0],  1.5);
    ASSERT_EQUAL(float_output[1],  3.0);
    ASSERT_EQUAL(float_output[2],  6.0);
    ASSERT_EQUAL(float_output[3], 10.0);
    
    // float -> int should use using plus<int> operator by default
    komrade::exclusive_scan(float_input.begin(), float_input.end(), int_output.begin());
    ASSERT_EQUAL(int_output[0], 0);
    ASSERT_EQUAL(int_output[1], 1);
    ASSERT_EQUAL(int_output[2], 3);
    ASSERT_EQUAL(int_output[3], 6);
    
    // float -> int should use using plus<int> operator by default
    komrade::exclusive_scan(float_input.begin(), float_input.end(), int_output.begin(), (float) 5.5);
    ASSERT_EQUAL(int_output[0],  5);
    ASSERT_EQUAL(int_output[1],  6);
    ASSERT_EQUAL(int_output[2],  8);
    ASSERT_EQUAL(int_output[3], 11);
    
    // int -> float should use using plus<float> operator by default
    komrade::inclusive_scan(int_input.begin(), int_input.end(), float_output.begin());
    ASSERT_EQUAL(float_output[0],  1.0);
    ASSERT_EQUAL(float_output[1],  3.0);
    ASSERT_EQUAL(float_output[2],  6.0);
    ASSERT_EQUAL(float_output[3], 10.0);
    
    // int -> float should use using plus<float> operator by default
    komrade::exclusive_scan(int_input.begin(), int_input.end(), float_output.begin(), (float) 5.5);
    ASSERT_EQUAL(float_output[0],  5.5);
    ASSERT_EQUAL(float_output[1],  6.5);
    ASSERT_EQUAL(float_output[2],  8.5);
    ASSERT_EQUAL(float_output[3], 11.5);
}
void TestScanMixedTypesHost(void)
{
    TestScanMixedTypes< komrade::host_vector<int>, komrade::host_vector<float> >();
}
DECLARE_UNITTEST(TestScanMixedTypesHost);
void TestScanMixedTypesDevice(void)
{
    TestScanMixedTypes< komrade::device_vector<int>, komrade::device_vector<float> >();
}
DECLARE_UNITTEST(TestScanMixedTypesDevice);


template <typename T>
void TestScan(const size_t n)
{
    komrade::host_vector<T>   h_input = komradetest::random_integers<T>(n);
    for(size_t i = 0; i < n; i++)
        h_input[i] = (((int) h_input[i]) % 81 - 40) / 4.0;  // floats will be XX.0, XX.25, XX.5, or XX.75
    komrade::device_vector<T> d_input = h_input;

    komrade::host_vector<T>   h_output(n);
    komrade::device_vector<T> d_output(n);
    
    komrade::host_vector<float>   h_float_output(n);
    komrade::device_vector<float> d_float_output(n);
    komrade::host_vector<int>   h_int_output(n);
    komrade::device_vector<int> d_int_output(n);

    
    komrade::inclusive_scan(h_input.begin(), h_input.end(), h_output.begin());
    komrade::inclusive_scan(d_input.begin(), d_input.end(), d_output.begin());
    ASSERT_EQUAL(d_output, h_output);
    
    komrade::exclusive_scan(h_input.begin(), h_input.end(), h_output.begin());
    komrade::exclusive_scan(d_input.begin(), d_input.end(), d_output.begin());
    ASSERT_EQUAL(d_output, h_output);
    
    komrade::exclusive_scan(h_input.begin(), h_input.end(), h_output.begin(), (T) 11);
    komrade::exclusive_scan(d_input.begin(), d_input.end(), d_output.begin(), (T) 11);
    ASSERT_EQUAL(d_output, h_output);
    
    //mixed input/output types
    komrade::inclusive_scan(h_input.begin(), h_input.end(), h_float_output.begin());
    komrade::inclusive_scan(d_input.begin(), d_input.end(), d_float_output.begin());
    ASSERT_EQUAL(d_output, h_output);
    
    komrade::exclusive_scan(h_input.begin(), h_input.end(), h_float_output.begin(), (float) 3.1415);
    komrade::exclusive_scan(d_input.begin(), d_input.end(), d_float_output.begin(), (float) 3.1415);
    ASSERT_EQUAL(d_output, h_output);
    
    komrade::exclusive_scan(h_input.begin(), h_input.end(), h_float_output.begin(), (int) 3);
    komrade::exclusive_scan(d_input.begin(), d_input.end(), d_float_output.begin(), (int) 3);
    ASSERT_EQUAL(d_output, h_output);
    
    komrade::exclusive_scan(h_input.begin(), h_input.end(), h_int_output.begin(), (int) 3);
    komrade::exclusive_scan(d_input.begin(), d_input.end(), d_int_output.begin(), (int) 3);
    ASSERT_EQUAL(d_output, h_output);
    
    komrade::exclusive_scan(h_input.begin(), h_input.end(), h_int_output.begin(), (float) 3.1415);
    komrade::exclusive_scan(d_input.begin(), d_input.end(), d_int_output.begin(), (float) 3.1415);
    ASSERT_EQUAL(d_output, h_output);
    
    // in-place scan
    komrade::inclusive_scan(h_input.begin(), h_input.end(), h_input.begin());
    komrade::inclusive_scan(d_input.begin(), d_input.end(), d_input.begin());
    ASSERT_EQUAL(d_input, h_input);
}
DECLARE_VARIABLE_UNITTEST(TestScan);


template <typename T>
struct TestInclusiveScanPair
{
  void operator()(const size_t n)
  {
    komradetest::random_integer<T> rnd;

    komrade::host_vector< komradetest::test_pair<T,T> > h_input(n);
    for(size_t i = 0; i < n; i++){
        h_input[i].first  = rnd();
        h_input[i].second = rnd();
    }
    komrade::device_vector< komradetest::test_pair<T,T> > d_input = h_input;
    
    komrade::host_vector< komradetest::test_pair<T,T> > h_output(n);
    komrade::device_vector< komradetest::test_pair<T,T> > d_output(n);

    //komradetest::test_pair<T,T> init(13, 29);

    komrade::inclusive_scan(h_input.begin(), h_input.end(), h_output.begin());
    komrade::inclusive_scan(d_input.begin(), d_input.end(), d_output.begin());

    ASSERT_EQUAL(d_input, h_input);
    ASSERT_EQUAL(d_output, h_output);
  }
};
VariableUnitTest<TestInclusiveScanPair, IntegralTypes> TestInclusiveScanPairInstance;

