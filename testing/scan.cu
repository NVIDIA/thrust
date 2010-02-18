#include <thrusttest/unittest.h>
#include <thrust/scan.h>
#include <thrust/functional.h>

template <class Vector>
void TestScanSimple(void)
{
    typedef typename Vector::value_type T;
    
    typename Vector::iterator iter;

    Vector input(5);
    Vector result(5);
    Vector output(5);

    input[0] = 1; input[1] = 3; input[2] = -2; input[3] = 4; input[4] = -5;

    Vector input_copy(input);

    // inclusive scan
    iter = thrust::inclusive_scan(input.begin(), input.end(), output.begin());
    result[0] = 1; result[1] = 4; result[2] = 2; result[3] = 6; result[4] = 1;
    ASSERT_EQUAL(iter - output.begin(), input.size());
    ASSERT_EQUAL(input,  input_copy);
    ASSERT_EQUAL(output, result);
    
    // exclusive scan
    iter = thrust::exclusive_scan(input.begin(), input.end(), output.begin(), 0);
    result[0] = 0; result[1] = 1; result[2] = 4; result[3] = 2; result[4] = 6;
    ASSERT_EQUAL(iter - output.begin(), input.size());
    ASSERT_EQUAL(input,  input_copy);
    ASSERT_EQUAL(output, result);
    
    // exclusive scan with init
    iter = thrust::exclusive_scan(input.begin(), input.end(), output.begin(), 3);
    result[0] = 3; result[1] = 4; result[2] = 7; result[3] = 5; result[4] = 9;
    ASSERT_EQUAL(iter - output.begin(), input.size());
    ASSERT_EQUAL(input,  input_copy);
    ASSERT_EQUAL(output, result);
    
    // inclusive scan with op
    iter = thrust::inclusive_scan(input.begin(), input.end(), output.begin(), thrust::plus<T>());
    result[0] = 1; result[1] = 4; result[2] = 2; result[3] = 6; result[4] = 1;
    ASSERT_EQUAL(iter - output.begin(), input.size());
    ASSERT_EQUAL(input,  input_copy);
    ASSERT_EQUAL(output, result);

    // exclusive scan with init and op
    iter = thrust::exclusive_scan(input.begin(), input.end(), output.begin(), 3, thrust::plus<T>());
    result[0] = 3; result[1] = 4; result[2] = 7; result[3] = 5; result[4] = 9;
    ASSERT_EQUAL(iter - output.begin(), input.size());
    ASSERT_EQUAL(input,  input_copy);
    ASSERT_EQUAL(output, result);

    // inplace inclusive scan
    input = input_copy;
    iter = thrust::inclusive_scan(input.begin(), input.end(), input.begin());
    result[0] = 1; result[1] = 4; result[2] = 2; result[3] = 6; result[4] = 1;
    ASSERT_EQUAL(iter - input.begin(), input.size());
    ASSERT_EQUAL(input, result);

    // inplace exclusive scan with init
    input = input_copy;
    iter = thrust::exclusive_scan(input.begin(), input.end(), input.begin(), 3);
    result[0] = 3; result[1] = 4; result[2] = 7; result[3] = 5; result[4] = 9;
    ASSERT_EQUAL(iter - input.begin(), input.size());
    ASSERT_EQUAL(input, result);

    // inplace exclusive scan with implicit init=0
    input = input_copy;
    iter = thrust::exclusive_scan(input.begin(), input.end(), input.begin());
    result[0] = 0; result[1] = 1; result[2] = 4; result[3] = 2; result[4] = 6;
    ASSERT_EQUAL(iter - input.begin(), input.size());
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
struct TestScanWithOperator
{
  void operator()(const size_t n)
  {
    thrust::host_vector<T>   h_input = thrusttest::random_integers<T>(n);
    thrust::device_vector<T> d_input = h_input;

    thrust::host_vector<T>   h_output(n);
    thrust::device_vector<T> d_output(n);
    
    thrust::inclusive_scan(h_input.begin(), h_input.end(), h_output.begin(), thrust::maximum<T>());
    thrust::inclusive_scan(d_input.begin(), d_input.end(), d_output.begin(), thrust::maximum<T>());
    ASSERT_EQUAL(d_output, h_output);
    
    thrust::exclusive_scan(h_input.begin(), h_input.end(), h_output.begin(), T(13), thrust::maximum<T>());
    thrust::exclusive_scan(d_input.begin(), d_input.end(), d_output.begin(), T(13), thrust::maximum<T>());
    ASSERT_EQUAL(d_output, h_output);
    }
};
VariableUnitTest<TestScanWithOperator, IntegralTypes> TestScanWithOperatorInstance;


template <typename T>
struct TestScan
{
  void operator()(const size_t n)
  {
    thrust::host_vector<T>   h_input = thrusttest::random_integers<T>(n);
    thrust::device_vector<T> d_input = h_input;

    thrust::host_vector<T>   h_output(n);
    thrust::device_vector<T> d_output(n);
    
    thrust::inclusive_scan(h_input.begin(), h_input.end(), h_output.begin());
    thrust::inclusive_scan(d_input.begin(), d_input.end(), d_output.begin());
    ASSERT_EQUAL(d_output, h_output);
    
    thrust::exclusive_scan(h_input.begin(), h_input.end(), h_output.begin());
    thrust::exclusive_scan(d_input.begin(), d_input.end(), d_output.begin());
    ASSERT_EQUAL(d_output, h_output);
    
    thrust::exclusive_scan(h_input.begin(), h_input.end(), h_output.begin(), (T) 11);
    thrust::exclusive_scan(d_input.begin(), d_input.end(), d_output.begin(), (T) 11);
    ASSERT_EQUAL(d_output, h_output);
    
    // in-place scans
    h_output = h_input;
    d_output = d_input;
    thrust::inclusive_scan(h_output.begin(), h_output.end(), h_output.begin());
    thrust::inclusive_scan(d_output.begin(), d_output.end(), d_output.begin());
    ASSERT_EQUAL(d_output, h_output);
    
    h_output = h_input;
    d_output = d_input;
    thrust::exclusive_scan(h_output.begin(), h_output.end(), h_output.begin());
    thrust::exclusive_scan(d_output.begin(), d_output.end(), d_output.begin());
    ASSERT_EQUAL(d_output, h_output);
    }
};
VariableUnitTest<TestScan, IntegralTypes> TestScanInstance;


void TestScanMixedTypes(void)
{
    const unsigned int n = 113;

    thrust::host_vector<unsigned int> h_input = thrusttest::random_integers<unsigned int>(n);
    for(size_t i = 0; i < n; i++)
        h_input[i] %= 10;
    thrust::device_vector<unsigned int> d_input = h_input;

    thrust::host_vector<float>   h_float_output(n);
    thrust::device_vector<float> d_float_output(n);
    thrust::host_vector<int>   h_int_output(n);
    thrust::device_vector<int> d_int_output(n);

    //mixed input/output types
    thrust::inclusive_scan(h_input.begin(), h_input.end(), h_float_output.begin());
    thrust::inclusive_scan(d_input.begin(), d_input.end(), d_float_output.begin());
    ASSERT_EQUAL(d_float_output, h_float_output);
    
    thrust::exclusive_scan(h_input.begin(), h_input.end(), h_float_output.begin(), (float) 3.5);
    thrust::exclusive_scan(d_input.begin(), d_input.end(), d_float_output.begin(), (float) 3.5);
    ASSERT_EQUAL(d_float_output, h_float_output);
    
    thrust::exclusive_scan(h_input.begin(), h_input.end(), h_float_output.begin(), (int) 3);
    thrust::exclusive_scan(d_input.begin(), d_input.end(), d_float_output.begin(), (int) 3);
    ASSERT_EQUAL(d_float_output, h_float_output);
    
    thrust::exclusive_scan(h_input.begin(), h_input.end(), h_int_output.begin(), (int) 3);
    thrust::exclusive_scan(d_input.begin(), d_input.end(), d_int_output.begin(), (int) 3);
    ASSERT_EQUAL(d_int_output, h_int_output);
    
    thrust::exclusive_scan(h_input.begin(), h_input.end(), h_int_output.begin(), (float) 3.5);
    thrust::exclusive_scan(d_input.begin(), d_input.end(), d_int_output.begin(), (float) 3.5);
    ASSERT_EQUAL(d_int_output, h_int_output);
}
DECLARE_UNITTEST(TestScanMixedTypes);


template <typename T, unsigned int N>
void _TestScanWithLargeTypes(void)
{
    size_t n = (64 * 1024) / sizeof(FixedVector<T,N>);

    thrust::host_vector< FixedVector<T,N> > h_input(n);
    thrust::host_vector< FixedVector<T,N> > h_output(n);

    for(size_t i = 0; i < h_input.size(); i++)
        h_input[i] = FixedVector<T,N>(i);

    thrust::device_vector< FixedVector<T,N> > d_input = h_input;
    thrust::device_vector< FixedVector<T,N> > d_output(n);
    
    thrust::inclusive_scan(h_input.begin(), h_input.end(), h_output.begin());
    thrust::inclusive_scan(d_input.begin(), d_input.end(), d_output.begin());

    ASSERT_EQUAL_QUIET(h_output, d_output);
    
    thrust::exclusive_scan(h_input.begin(), h_input.end(), h_output.begin(), FixedVector<T,N>(0));
    thrust::exclusive_scan(d_input.begin(), d_input.end(), d_output.begin(), FixedVector<T,N>(0));
    
    ASSERT_EQUAL_QUIET(h_output, d_output);
}

void TestScanWithLargeTypes(void)
{
#ifdef _MSC_VER
    // XXX when compiling, these tests kill the backend for some reason on XP
    KNOWN_FAILURE
#else
    _TestScanWithLargeTypes<int,    1>();
    _TestScanWithLargeTypes<int,    2>();
    _TestScanWithLargeTypes<int,    4>();
    _TestScanWithLargeTypes<int,    8>();
    _TestScanWithLargeTypes<int,   16>();
    _TestScanWithLargeTypes<int,   32>();
    _TestScanWithLargeTypes<int,   64>();
#endif

    //_TestScanWithLargeTypes<int,  128>();
    //_TestScanWithLargeTypes<int,  256>();
    //_TestScanWithLargeTypes<int,  512>();
    //_TestScanWithLargeTypes<int, 1024>();
}
DECLARE_UNITTEST(TestScanWithLargeTypes);


template <typename T>
struct plus_mod3 : public thrust::binary_function<T,T,T>
{
    T * table;

    plus_mod3(T * table) : table(table) {}

    __host__ __device__
    T operator()(T a, T b)
    {
        return table[(int) (a + b)];
    }
};

template <typename Vector>
void TestInclusiveScanWithIndirection(void)
{
    // add numbers modulo 3 with external lookup table
    typedef typename Vector::value_type T;

    Vector data(7);
    data[0] = 0;
    data[1] = 1;
    data[2] = 2;
    data[3] = 1;
    data[4] = 2;
    data[5] = 0;
    data[6] = 1;

    Vector table(6);
    table[0] = 0;
    table[1] = 1;
    table[2] = 2;
    table[3] = 0;
    table[4] = 1;
    table[5] = 2;

    thrust::inclusive_scan(data.begin(), data.end(), data.begin(), plus_mod3<T>(thrust::raw_pointer_cast(&table[0])));
    
    ASSERT_EQUAL(data[0], T(0));
    ASSERT_EQUAL(data[1], T(1));
    ASSERT_EQUAL(data[2], T(0));
    ASSERT_EQUAL(data[3], T(1));
    ASSERT_EQUAL(data[4], T(0));
    ASSERT_EQUAL(data[5], T(0));
    ASSERT_EQUAL(data[6], T(1));
}
DECLARE_VECTOR_UNITTEST(TestInclusiveScanWithIndirection);

