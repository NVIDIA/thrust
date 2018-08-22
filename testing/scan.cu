#include <unittest/unittest.h>
#include <thrust/scan.h>
#include <thrust/functional.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/retag.h>


template<typename T>
  struct max_functor
{
  __host__ __device__
  T operator()(T rhs, T lhs) const
  {
    return thrust::max(rhs,lhs);
  }
};


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
    ASSERT_EQUAL(std::size_t(iter - output.begin()), input.size());
    ASSERT_EQUAL(input,  input_copy);
    ASSERT_EQUAL(output, result);
    
    // exclusive scan
    iter = thrust::exclusive_scan(input.begin(), input.end(), output.begin(), T(0));
    result[0] = 0; result[1] = 1; result[2] = 4; result[3] = 2; result[4] = 6;
    ASSERT_EQUAL(std::size_t(iter - output.begin()), input.size());
    ASSERT_EQUAL(input,  input_copy);
    ASSERT_EQUAL(output, result);
    
    // exclusive scan with init
    iter = thrust::exclusive_scan(input.begin(), input.end(), output.begin(), T(3));
    result[0] = 3; result[1] = 4; result[2] = 7; result[3] = 5; result[4] = 9;
    ASSERT_EQUAL(std::size_t(iter - output.begin()), input.size());
    ASSERT_EQUAL(input,  input_copy);
    ASSERT_EQUAL(output, result);
    
    // inclusive scan with op
    iter = thrust::inclusive_scan(input.begin(), input.end(), output.begin(), thrust::plus<T>());
    result[0] = 1; result[1] = 4; result[2] = 2; result[3] = 6; result[4] = 1;
    ASSERT_EQUAL(std::size_t(iter - output.begin()), input.size());
    ASSERT_EQUAL(input,  input_copy);
    ASSERT_EQUAL(output, result);

    // exclusive scan with init and op
    iter = thrust::exclusive_scan(input.begin(), input.end(), output.begin(), T(3), thrust::plus<T>());
    result[0] = 3; result[1] = 4; result[2] = 7; result[3] = 5; result[4] = 9;
    ASSERT_EQUAL(std::size_t(iter - output.begin()), input.size());
    ASSERT_EQUAL(input,  input_copy);
    ASSERT_EQUAL(output, result);

    // inplace inclusive scan
    input = input_copy;
    iter = thrust::inclusive_scan(input.begin(), input.end(), input.begin());
    result[0] = 1; result[1] = 4; result[2] = 2; result[3] = 6; result[4] = 1;
    ASSERT_EQUAL(std::size_t(iter - input.begin()), input.size());
    ASSERT_EQUAL(input, result);

    // inplace exclusive scan with init
    input = input_copy;
    iter = thrust::exclusive_scan(input.begin(), input.end(), input.begin(), T(3));
    result[0] = 3; result[1] = 4; result[2] = 7; result[3] = 5; result[4] = 9;
    ASSERT_EQUAL(std::size_t(iter - input.begin()), input.size());
    ASSERT_EQUAL(input, result);

    // inplace exclusive scan with implicit init=0
    input = input_copy;
    iter = thrust::exclusive_scan(input.begin(), input.end(), input.begin());
    result[0] = 0; result[1] = 1; result[2] = 4; result[3] = 2; result[4] = 6;
    ASSERT_EQUAL(std::size_t(iter - input.begin()), input.size());
    ASSERT_EQUAL(input, result);
}
DECLARE_VECTOR_UNITTEST(TestScanSimple);


template<typename InputIterator,
         typename OutputIterator>
OutputIterator inclusive_scan(my_system &system,
                              InputIterator,
                              InputIterator,
                              OutputIterator result)
{
    system.validate_dispatch();
    return result;
}

void TestInclusiveScanDispatchExplicit()
{
    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::inclusive_scan(sys,
                           vec.begin(),
                           vec.begin(),
                           vec.begin());

    ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestInclusiveScanDispatchExplicit);


template<typename InputIterator,
         typename OutputIterator>
OutputIterator inclusive_scan(my_tag,
                              InputIterator,
                              InputIterator,
                              OutputIterator result)
{
    *result = 13;
    return result;
}

void TestInclusiveScanDispatchImplicit()
{
    thrust::device_vector<int> vec(1);

    thrust::inclusive_scan(thrust::retag<my_tag>(vec.begin()),
                           thrust::retag<my_tag>(vec.begin()),
                           thrust::retag<my_tag>(vec.begin()));

    ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestInclusiveScanDispatchImplicit);


template<typename InputIterator,
         typename OutputIterator>
OutputIterator exclusive_scan(my_system &system,
                              InputIterator,
                              InputIterator,
                              OutputIterator result)
{
    system.validate_dispatch();
    return result;
}

void TestExclusiveScanDispatchExplicit()
{
    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::exclusive_scan(sys,
                           vec.begin(),
                           vec.begin(),
                           vec.begin());

    ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestExclusiveScanDispatchExplicit);


template<typename InputIterator,
         typename OutputIterator>
OutputIterator exclusive_scan(my_tag,
                              InputIterator,
                              InputIterator,
                              OutputIterator result)
{
    *result = 13;
    return result;
}

void TestExclusiveScanDispatchImplicit()
{
    thrust::device_vector<int> vec(1);

    thrust::exclusive_scan(thrust::retag<my_tag>(vec.begin()),
                           thrust::retag<my_tag>(vec.begin()),
                           thrust::retag<my_tag>(vec.begin()));

    ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestExclusiveScanDispatchImplicit);


void TestInclusiveScan32(void)
{
    typedef int T;
    size_t n = 32;

    thrust::host_vector<T>   h_input = unittest::random_integers<T>(n);
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

    thrust::host_vector<T>   h_input = unittest::random_integers<T>(n);
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
    
    // float -> float with plus<int> operator (int accumulator)
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
    ASSERT_EQUAL(int_output[1],  7);
    ASSERT_EQUAL(int_output[2],  9);
    ASSERT_EQUAL(int_output[3], 13);
    
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
    thrust::host_vector<T>   h_input = unittest::random_integers<T>(n);
    thrust::device_vector<T> d_input = h_input;

    thrust::host_vector<T>   h_output(n);
    thrust::device_vector<T> d_output(n);
    
    thrust::inclusive_scan(h_input.begin(), h_input.end(), h_output.begin(), max_functor<T>());
    thrust::inclusive_scan(d_input.begin(), d_input.end(), d_output.begin(), max_functor<T>());
    ASSERT_EQUAL(d_output, h_output);
    
    thrust::exclusive_scan(h_input.begin(), h_input.end(), h_output.begin(), T(13), max_functor<T>());
    thrust::exclusive_scan(d_input.begin(), d_input.end(), d_output.begin(), T(13), max_functor<T>());
    ASSERT_EQUAL(d_output, h_output);
  }
};
VariableUnitTest<TestScanWithOperator, SignedIntegralTypes> TestScanWithOperatorInstance;


template <typename T>
struct TestScanWithOperatorToDiscardIterator
{
  void operator()(const size_t n)
  {
    thrust::host_vector<T>   h_input = unittest::random_integers<T>(n);
    thrust::device_vector<T> d_input = h_input;

    thrust::discard_iterator<> reference(n);
    
    thrust::discard_iterator<> h_result =
      thrust::inclusive_scan(h_input.begin(), h_input.end(), thrust::make_discard_iterator(), max_functor<T>());

    thrust::discard_iterator<> d_result =
      thrust::inclusive_scan(d_input.begin(), d_input.end(), thrust::make_discard_iterator(), max_functor<T>());
    
    ASSERT_EQUAL_QUIET(reference, h_result);
    ASSERT_EQUAL_QUIET(reference, d_result);
    
    h_result =
      thrust::exclusive_scan(h_input.begin(), h_input.end(), thrust::make_discard_iterator(), T(13), max_functor<T>());

    d_result =
      thrust::exclusive_scan(d_input.begin(), d_input.end(), thrust::make_discard_iterator(), T(13), max_functor<T>());

    ASSERT_EQUAL_QUIET(reference, h_result);
    ASSERT_EQUAL_QUIET(reference, d_result);
  }
};
VariableUnitTest<TestScanWithOperatorToDiscardIterator, unittest::type_list<unittest::int8_t,unittest::int16_t,unittest::int32_t> > TestScanWithOperatorToDiscardIteratorInstance;


template <typename T>
struct TestScan
{
  void operator()(const size_t n)
  {
    thrust::host_vector<T>   h_input = unittest::random_integers<T>(n);
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


template <typename T>
struct TestScanToDiscardIterator
{
  void operator()(const size_t n)
  {
    thrust::host_vector<T>   h_input = unittest::random_integers<T>(n);
    thrust::device_vector<T> d_input = h_input;
    
    thrust::discard_iterator<> h_result =
      thrust::inclusive_scan(h_input.begin(), h_input.end(), thrust::make_discard_iterator());

    thrust::discard_iterator<> d_result =
      thrust::inclusive_scan(d_input.begin(), d_input.end(), thrust::make_discard_iterator());

    thrust::discard_iterator<> reference(n);

    ASSERT_EQUAL_QUIET(reference, h_result);
    ASSERT_EQUAL_QUIET(reference, d_result);
    
    h_result =
      thrust::exclusive_scan(h_input.begin(), h_input.end(), thrust::make_discard_iterator(), (T) 11);

    d_result =
      thrust::exclusive_scan(d_input.begin(), d_input.end(), thrust::make_discard_iterator(), (T) 11);

    ASSERT_EQUAL_QUIET(reference, h_result);
    ASSERT_EQUAL_QUIET(reference, d_result);
  }
};
VariableUnitTest<TestScanToDiscardIterator, unittest::type_list<unittest::int8_t,unittest::int16_t,unittest::int32_t> > TestScanToDiscardIteratorInstance;


void TestScanMixedTypes(void)
{
    const unsigned int n = 113;

    thrust::host_vector<unsigned int> h_input = unittest::random_integers<unsigned int>(n);
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
    size_t n = (1024 * 1024) / sizeof(FixedVector<T,N>);

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
  _TestScanWithLargeTypes<int,  1>();

#if !defined(__QNX__)
  _TestScanWithLargeTypes<int,  8>();
  _TestScanWithLargeTypes<int, 64>();
#else
  KNOWN_FAILURE;
#endif
}
DECLARE_UNITTEST(TestScanWithLargeTypes);


template <typename T>
struct plus_mod3
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
DECLARE_INTEGRAL_VECTOR_UNITTEST(TestInclusiveScanWithIndirection);

