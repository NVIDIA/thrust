#include <unittest/unittest.h>
#include <thrust/range/algorithm/reduce.h>

//template<typename T>
//struct is_equal_div_10_reduce
//{
//    __host__ __device__
//    bool operator()(const T x, const T& y) const { return ((int) x / 10) == ((int) y / 10); }
//};

template <class Vector>
void TestRangeReduceSimple(void)
{
    typedef typename Vector::value_type T;

    Vector v(3);
    v[0] = 1; v[1] = -2; v[2] = 3;

    // no initializer
    ASSERT_EQUAL(2, thrust::experimental::range::reduce(v));

    // with initializer
    ASSERT_EQUAL(12, thrust::experimental::range::reduce(v, (T) 10));
}
DECLARE_VECTOR_UNITTEST(TestRangeReduceSimple);


template <typename T>
void TestRangeReduce(const size_t n)
{
    thrust::host_vector<T>   h_data = unittest::random_integers<T>(n);
    thrust::device_vector<T> d_data = h_data;

    T init = 13;

    T cpu_result = thrust::experimental::range::reduce(h_data, init);
    T gpu_result = thrust::experimental::range::reduce(d_data, init);

    ASSERT_ALMOST_EQUAL(cpu_result, gpu_result);
}
DECLARE_VARIABLE_UNITTEST(TestRangeReduce);


template <class IntVector, class FloatVector>
void TestRangeReduceMixedTypes(void)
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

    // float -> int should use using plus<int> operator by default
    ASSERT_EQUAL(10, thrust::experimental::range::reduce(float_input, (int) 0));
    
    // int -> float should use using plus<float> operator by default
    ASSERT_EQUAL(10.5, thrust::experimental::range::reduce(int_input, (float) 0.5));
}
void TestRangeReduceMixedTypesHost(void)
{
    TestRangeReduceMixedTypes< thrust::host_vector<int>, thrust::host_vector<float> >();
}
DECLARE_UNITTEST(TestRangeReduceMixedTypesHost);
void TestRangeReduceMixedTypesDevice(void)
{
    TestRangeReduceMixedTypes< thrust::device_vector<int>, thrust::device_vector<float> >();
}
DECLARE_UNITTEST(TestRangeReduceMixedTypesDevice);



template <typename T>
void TestRangeReduceWithOperator(const size_t n)
{
    thrust::host_vector<T>   h_data = unittest::random_integers<T>(n);
    thrust::device_vector<T> d_data = h_data;

    T init = 0;

    T cpu_result = thrust::experimental::range::reduce(h_data, init, thrust::maximum<T>());
    T gpu_result = thrust::experimental::range::reduce(d_data, init, thrust::maximum<T>());

    ASSERT_EQUAL(cpu_result, gpu_result);
}
DECLARE_VARIABLE_UNITTEST(TestRangeReduceWithOperator);


template <typename T, unsigned int N>
void _TestRangeReduceWithLargeTypes(void)
{
    size_t n = (64 * 1024) / sizeof(FixedVector<T,N>);

    thrust::host_vector< FixedVector<T,N> > h_data(n);

    for(size_t i = 0; i < h_data.size(); i++)
        h_data[i] = FixedVector<T,N>(i);

    thrust::device_vector< FixedVector<T,N> > d_data = h_data;
    
    FixedVector<T,N> h_result = thrust::experimental::range::reduce(h_data, FixedVector<T,N>(0));
    FixedVector<T,N> d_result = thrust::experimental::range::reduce(d_data, FixedVector<T,N>(0));

    ASSERT_EQUAL_QUIET(h_result, d_result);
}

void TestRangeReduceWithLargeTypes(void)
{
    _TestRangeReduceWithLargeTypes<int,    1>();
    _TestRangeReduceWithLargeTypes<int,    2>();
    _TestRangeReduceWithLargeTypes<int,    4>();
    _TestRangeReduceWithLargeTypes<int,    8>();
    _TestRangeReduceWithLargeTypes<int,   16>();
    _TestRangeReduceWithLargeTypes<int,   32>();
    _TestRangeReduceWithLargeTypes<int,   64>();
    _TestRangeReduceWithLargeTypes<int,  128>(); 
    _TestRangeReduceWithLargeTypes<int,  256>(); 
    _TestRangeReduceWithLargeTypes<int,  512>();
    //_TestRangeReduceWithLargeTypes<int, 1024>(); // out of memory
    //_TestRangeReduceWithLargeTypes<int, 2048>(); // uses too much local data
    //_TestRangeReduceWithLargeTypes<int, 4096>(); // uses too much local data
}
DECLARE_UNITTEST(TestRangeReduceWithLargeTypes);

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
void TestRangeReduceWithIndirection(void)
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

    T result = thrust::experimental::range::reduce(data, T(0), plus_mod3<T>(thrust::raw_pointer_cast(&table[0])));
    
    ASSERT_EQUAL(T(1), result);
}
DECLARE_VECTOR_UNITTEST(TestRangeReduceWithIndirection);



//template <typename Vector>
//void initialize_keys(Vector& keys)
//{
//    keys.resize(9);
//    keys[0] = 11;
//    keys[1] = 11;
//    keys[2] = 21;
//    keys[3] = 20;
//    keys[4] = 21;
//    keys[5] = 21;
//    keys[6] = 21;
//    keys[7] = 37;
//    keys[8] = 37;
//}
//
//template <typename Vector>
//void initialize_values(Vector& values)
//{
//    values.resize(9);
//    values[0] = 0; 
//    values[1] = 1;
//    values[2] = 2;
//    values[3] = 3;
//    values[4] = 4;
//    values[5] = 5;
//    values[6] = 6;
//    values[7] = 7;
//    values[8] = 8;
//}
//
//
//template<typename Vector>
//void TestReduceByKeySimple(void)
//{
//    typedef typename Vector::value_type T;
//
//    Vector keys;
//    Vector values;
//
//    typename thrust::pair<typename Vector::iterator, typename Vector::iterator> new_last;
//
//    // basic test
//    initialize_keys(keys);  initialize_values(values);
//
//    Vector output_keys(keys.size());
//    Vector output_values(values.size());
//
//    new_last = thrust::reduce_by_key(keys.begin(), keys.end(), values.begin(), output_keys.begin(), output_values.begin());
//
//    ASSERT_EQUAL(new_last.first  - output_keys.begin(),   5);
//    ASSERT_EQUAL(new_last.second - output_values.begin(), 5);
//    ASSERT_EQUAL(output_keys[0], 11);
//    ASSERT_EQUAL(output_keys[1], 21);
//    ASSERT_EQUAL(output_keys[2], 20);
//    ASSERT_EQUAL(output_keys[3], 21);
//    ASSERT_EQUAL(output_keys[4], 37);
//    
//    ASSERT_EQUAL(output_values[0],  1);
//    ASSERT_EQUAL(output_values[1],  2);
//    ASSERT_EQUAL(output_values[2],  3);
//    ASSERT_EQUAL(output_values[3], 15);
//    ASSERT_EQUAL(output_values[4], 15);
//
//    // test BinaryPredicate
//    initialize_keys(keys);  initialize_values(values);
//    
//    new_last = thrust::reduce_by_key(keys.begin(), keys.end(), values.begin(), output_keys.begin(), output_values.begin(), is_equal_div_10_reduce<T>());
//
//    ASSERT_EQUAL(new_last.first  - output_keys.begin(),   3);
//    ASSERT_EQUAL(new_last.second - output_values.begin(), 3);
//    ASSERT_EQUAL(output_keys[0], 11);
//    ASSERT_EQUAL(output_keys[1], 21);
//    ASSERT_EQUAL(output_keys[2], 37);
//    
//    ASSERT_EQUAL(output_values[0],  1);
//    ASSERT_EQUAL(output_values[1], 20);
//    ASSERT_EQUAL(output_values[2], 15);
//
//    // test BinaryFunction
//    initialize_keys(keys);  initialize_values(values);
//
//    new_last = thrust::reduce_by_key(keys.begin(), keys.end(), values.begin(), output_keys.begin(), output_values.begin(), thrust::equal_to<T>(), thrust::plus<T>());
//
//    ASSERT_EQUAL(new_last.first  - output_keys.begin(),   5);
//    ASSERT_EQUAL(new_last.second - output_values.begin(), 5);
//    ASSERT_EQUAL(output_keys[0], 11);
//    ASSERT_EQUAL(output_keys[1], 21);
//    ASSERT_EQUAL(output_keys[2], 20);
//    ASSERT_EQUAL(output_keys[3], 21);
//    ASSERT_EQUAL(output_keys[4], 37);
//    
//    ASSERT_EQUAL(output_values[0],  1);
//    ASSERT_EQUAL(output_values[1],  2);
//    ASSERT_EQUAL(output_values[2],  3);
//    ASSERT_EQUAL(output_values[3], 15);
//    ASSERT_EQUAL(output_values[4], 15);
//}
//DECLARE_VECTOR_UNITTEST(TestReduceByKeySimple);
//
//template<typename K>
//struct TestReduceByKey
//{
//    void operator()(const size_t n)
//    {
//        typedef unsigned int V; // ValueType
//
//        thrust::host_vector<K>   h_keys = unittest::random_integers<bool>(n);
//        thrust::host_vector<V>   h_vals = unittest::random_integers<V>(n);
//        thrust::device_vector<K> d_keys = h_keys;
//        thrust::device_vector<V> d_vals = h_vals;
//
//        thrust::host_vector<K>   h_keys_output(n);
//        thrust::host_vector<V>   h_vals_output(n);
//        thrust::device_vector<K> d_keys_output(n);
//        thrust::device_vector<V> d_vals_output(n);
//
//        typedef typename thrust::host_vector<K>::iterator   HostKeyIterator;
//        typedef typename thrust::host_vector<V>::iterator   HostValIterator;
//        typedef typename thrust::device_vector<K>::iterator DeviceKeyIterator;
//        typedef typename thrust::device_vector<V>::iterator DeviceValIterator;
//
//        typedef typename thrust::pair<HostKeyIterator,  HostValIterator>   HostIteratorPair;
//        typedef typename thrust::pair<DeviceKeyIterator,DeviceValIterator> DeviceIteratorPair;
//
//        HostIteratorPair   h_last = thrust::reduce_by_key(h_keys.begin(), h_keys.end(), h_vals.begin(), h_keys_output.begin(), h_vals_output.begin());
//        DeviceIteratorPair d_last = thrust::reduce_by_key(d_keys.begin(), d_keys.end(), d_vals.begin(), d_keys_output.begin(), d_vals_output.begin());
//
//        ASSERT_EQUAL(h_last.first  - h_keys_output.begin(), d_last.first  - d_keys_output.begin());
//        ASSERT_EQUAL(h_last.second - h_vals_output.begin(), d_last.second - d_vals_output.begin());
//       
//        size_t N = h_last.first - h_keys_output.begin();
//
//        h_keys_output.resize(N);
//        h_vals_output.resize(N);
//        d_keys_output.resize(N);
//        d_vals_output.resize(N);
//
//        ASSERT_EQUAL(h_keys_output, d_keys_output);
//        ASSERT_EQUAL(h_vals_output, d_vals_output);
//    }
//};
//VariableUnitTest<TestReduceByKey, IntegralTypes> TestReduceByKeyInstance;
//
//
