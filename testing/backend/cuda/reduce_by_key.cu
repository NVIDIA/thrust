#include <unittest/unittest.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>


template<typename ExecutionPolicy, typename Iterator1, typename Iterator2, typename Iterator3, typename Iterator4, typename Iterator5>
__global__
void reduce_by_key_kernel(ExecutionPolicy exec,
                          Iterator1 keys_first, Iterator1 keys_last,
                          Iterator2 values_first,
                          Iterator3 keys_result,
                          Iterator4 values_result,
                          Iterator5 result)
{
  *result = thrust::reduce_by_key(exec, keys_first, keys_last, values_first, keys_result, values_result);
}


template<typename ExecutionPolicy, typename Iterator1, typename Iterator2, typename Iterator3, typename Iterator4, typename BinaryPredicate, typename Iterator5>
__global__
void reduce_by_key_kernel(ExecutionPolicy exec,
                          Iterator1 keys_first, Iterator1 keys_last,
                          Iterator2 values_first,
                          Iterator3 keys_result,
                          Iterator4 values_result,
                          BinaryPredicate pred,
                          Iterator5 result)
{
  *result = thrust::reduce_by_key(exec, keys_first, keys_last, values_first, keys_result, values_result, pred);
}


template<typename ExecutionPolicy, typename Iterator1, typename Iterator2, typename Iterator3, typename Iterator4, typename BinaryPredicate, typename BinaryFunction, typename Iterator5>
__global__
void reduce_by_key_kernel(ExecutionPolicy exec,
                          Iterator1 keys_first, Iterator1 keys_last,
                          Iterator2 values_first,
                          Iterator3 keys_result,
                          Iterator4 values_result,
                          BinaryPredicate pred,
                          BinaryFunction binary_op,
                          Iterator5 result)
{
  *result = thrust::reduce_by_key(exec, keys_first, keys_last, values_first, keys_result, values_result, pred, binary_op);
}


template<typename T>
struct is_equal_div_10_reduce
{
  __host__ __device__
  bool operator()(const T x, const T& y) const { return ((int) x / 10) == ((int) y / 10); }
};


template<typename Vector>
void initialize_keys(Vector& keys)
{
  keys.resize(9);
  keys[0] = 11;
  keys[1] = 11;
  keys[2] = 21;
  keys[3] = 20;
  keys[4] = 21;
  keys[5] = 21;
  keys[6] = 21;
  keys[7] = 37;
  keys[8] = 37;
}


template<typename Vector>
void initialize_values(Vector& values)
{
  values.resize(9);
  values[0] = 0; 
  values[1] = 1;
  values[2] = 2;
  values[3] = 3;
  values[4] = 4;
  values[5] = 5;
  values[6] = 6;
  values[7] = 7;
  values[8] = 8;
}


void TestReduceByKeyDeviceSeq()
{
  typedef int T;
  
  thrust::device_vector<T> keys;
  thrust::device_vector<T> values;

  typedef typename thrust::pair<
    typename thrust::device_vector<T>::iterator,
    typename thrust::device_vector<T>::iterator
  > iterator_pair;

  thrust::device_vector<iterator_pair> new_last_vec(1);
  iterator_pair new_last;
  
  // basic test
  initialize_keys(keys);  initialize_values(values);
  
  thrust::device_vector<T> output_keys(keys.size());
  thrust::device_vector<T> output_values(values.size());
  
  reduce_by_key_kernel<<<1,1>>>(thrust::seq, keys.begin(), keys.end(), values.begin(), output_keys.begin(), output_values.begin(), new_last_vec.begin());
  new_last = new_last_vec[0];
  
  ASSERT_EQUAL(new_last.first  - output_keys.begin(),   5);
  ASSERT_EQUAL(new_last.second - output_values.begin(), 5);
  ASSERT_EQUAL(output_keys[0], 11);
  ASSERT_EQUAL(output_keys[1], 21);
  ASSERT_EQUAL(output_keys[2], 20);
  ASSERT_EQUAL(output_keys[3], 21);
  ASSERT_EQUAL(output_keys[4], 37);
  
  ASSERT_EQUAL(output_values[0],  1);
  ASSERT_EQUAL(output_values[1],  2);
  ASSERT_EQUAL(output_values[2],  3);
  ASSERT_EQUAL(output_values[3], 15);
  ASSERT_EQUAL(output_values[4], 15);
  
  // test BinaryPredicate
  initialize_keys(keys);  initialize_values(values);
  
  reduce_by_key_kernel<<<1,1>>>(thrust::seq, keys.begin(), keys.end(), values.begin(), output_keys.begin(), output_values.begin(), is_equal_div_10_reduce<T>(), new_last_vec.begin());
  new_last = new_last_vec[0];
  
  ASSERT_EQUAL(new_last.first  - output_keys.begin(),   3);
  ASSERT_EQUAL(new_last.second - output_values.begin(), 3);
  ASSERT_EQUAL(output_keys[0], 11);
  ASSERT_EQUAL(output_keys[1], 21);
  ASSERT_EQUAL(output_keys[2], 37);
  
  ASSERT_EQUAL(output_values[0],  1);
  ASSERT_EQUAL(output_values[1], 20);
  ASSERT_EQUAL(output_values[2], 15);
  
  // test BinaryFunction
  initialize_keys(keys);  initialize_values(values);
  
  reduce_by_key_kernel<<<1,1>>>(thrust::seq, keys.begin(), keys.end(), values.begin(), output_keys.begin(), output_values.begin(), thrust::equal_to<T>(), thrust::plus<T>(), new_last_vec.begin());
  new_last = new_last_vec[0];
  
  ASSERT_EQUAL(new_last.first  - output_keys.begin(),   5);
  ASSERT_EQUAL(new_last.second - output_values.begin(), 5);
  ASSERT_EQUAL(output_keys[0], 11);
  ASSERT_EQUAL(output_keys[1], 21);
  ASSERT_EQUAL(output_keys[2], 20);
  ASSERT_EQUAL(output_keys[3], 21);
  ASSERT_EQUAL(output_keys[4], 37);
  
  ASSERT_EQUAL(output_values[0],  1);
  ASSERT_EQUAL(output_values[1],  2);
  ASSERT_EQUAL(output_values[2],  3);
  ASSERT_EQUAL(output_values[3], 15);
  ASSERT_EQUAL(output_values[4], 15);
}
DECLARE_UNITTEST(TestReduceByKeyDeviceSeq);


void TestReduceByKeySimpleDeviceDevice()
{
  typedef int T;
  
  thrust::device_vector<T> keys;
  thrust::device_vector<T> values;

  typedef typename thrust::pair<
    typename thrust::device_vector<T>::iterator,
    typename thrust::device_vector<T>::iterator
  > iterator_pair;

  thrust::device_vector<iterator_pair> new_last_vec(1);
  iterator_pair new_last;
  
  // basic test
  initialize_keys(keys);  initialize_values(values);
  
  thrust::device_vector<T> output_keys(keys.size());
  thrust::device_vector<T> output_values(values.size());
  
  reduce_by_key_kernel<<<1,1>>>(thrust::device, keys.begin(), keys.end(), values.begin(), output_keys.begin(), output_values.begin(), new_last_vec.begin());
  new_last = new_last_vec[0];
  
  ASSERT_EQUAL(new_last.first  - output_keys.begin(),   5);
  ASSERT_EQUAL(new_last.second - output_values.begin(), 5);
  ASSERT_EQUAL(output_keys[0], 11);
  ASSERT_EQUAL(output_keys[1], 21);
  ASSERT_EQUAL(output_keys[2], 20);
  ASSERT_EQUAL(output_keys[3], 21);
  ASSERT_EQUAL(output_keys[4], 37);
  
  ASSERT_EQUAL(output_values[0],  1);
  ASSERT_EQUAL(output_values[1],  2);
  ASSERT_EQUAL(output_values[2],  3);
  ASSERT_EQUAL(output_values[3], 15);
  ASSERT_EQUAL(output_values[4], 15);
  
  // test BinaryPredicate
  initialize_keys(keys);  initialize_values(values);
  
  reduce_by_key_kernel<<<1,1>>>(thrust::device, keys.begin(), keys.end(), values.begin(), output_keys.begin(), output_values.begin(), is_equal_div_10_reduce<T>(), new_last_vec.begin());
  new_last = new_last_vec[0];
  
  ASSERT_EQUAL(new_last.first  - output_keys.begin(),   3);
  ASSERT_EQUAL(new_last.second - output_values.begin(), 3);
  ASSERT_EQUAL(output_keys[0], 11);
  ASSERT_EQUAL(output_keys[1], 21);
  ASSERT_EQUAL(output_keys[2], 37);
  
  ASSERT_EQUAL(output_values[0],  1);
  ASSERT_EQUAL(output_values[1], 20);
  ASSERT_EQUAL(output_values[2], 15);
  
  // test BinaryFunction
  initialize_keys(keys);  initialize_values(values);
  
  reduce_by_key_kernel<<<1,1>>>(thrust::device, keys.begin(), keys.end(), values.begin(), output_keys.begin(), output_values.begin(), thrust::equal_to<T>(), thrust::plus<T>(), new_last_vec.begin());
  new_last = new_last_vec[0];
  
  ASSERT_EQUAL(new_last.first  - output_keys.begin(),   5);
  ASSERT_EQUAL(new_last.second - output_values.begin(), 5);
  ASSERT_EQUAL(output_keys[0], 11);
  ASSERT_EQUAL(output_keys[1], 21);
  ASSERT_EQUAL(output_keys[2], 20);
  ASSERT_EQUAL(output_keys[3], 21);
  ASSERT_EQUAL(output_keys[4], 37);
  
  ASSERT_EQUAL(output_values[0],  1);
  ASSERT_EQUAL(output_values[1],  2);
  ASSERT_EQUAL(output_values[2],  3);
  ASSERT_EQUAL(output_values[3], 15);
  ASSERT_EQUAL(output_values[4], 15);
}
DECLARE_UNITTEST(TestReduceByKeySimpleDeviceDevice);


template<typename K>
struct TestReduceByKeyDeviceDevice
{
  void operator()(const size_t n)
  {
    typedef unsigned int V; // ValueType
    
    thrust::host_vector<K>   h_keys = unittest::random_integers<bool>(n);
    thrust::host_vector<V>   h_vals = unittest::random_integers<V>(n);
    thrust::device_vector<K> d_keys = h_keys;
    thrust::device_vector<V> d_vals = h_vals;
    
    thrust::host_vector<K>   h_keys_output(n);
    thrust::host_vector<V>   h_vals_output(n);
    thrust::device_vector<K> d_keys_output(n);
    thrust::device_vector<V> d_vals_output(n);
    
    typedef typename thrust::host_vector<K>::iterator   HostKeyIterator;
    typedef typename thrust::host_vector<V>::iterator   HostValIterator;
    typedef typename thrust::device_vector<K>::iterator DeviceKeyIterator;
    typedef typename thrust::device_vector<V>::iterator DeviceValIterator;
    
    typedef typename thrust::pair<HostKeyIterator,  HostValIterator>   HostIteratorPair;
    typedef typename thrust::pair<DeviceKeyIterator,DeviceValIterator> DeviceIteratorPair;
    
    HostIteratorPair   h_last = thrust::reduce_by_key(h_keys.begin(), h_keys.end(), h_vals.begin(), h_keys_output.begin(), h_vals_output.begin());

    thrust::device_vector<DeviceIteratorPair> d_last_vec(1);
    reduce_by_key_kernel<<<1,1>>>(thrust::device, d_keys.begin(), d_keys.end(), d_vals.begin(), d_keys_output.begin(), d_vals_output.begin(), d_last_vec.begin());
    DeviceIteratorPair d_last = d_last_vec[0];
    
    ASSERT_EQUAL(h_last.first  - h_keys_output.begin(), d_last.first  - d_keys_output.begin());
    ASSERT_EQUAL(h_last.second - h_vals_output.begin(), d_last.second - d_vals_output.begin());
    
    size_t N = h_last.first - h_keys_output.begin();
    
    h_keys_output.resize(N);
    h_vals_output.resize(N);
    d_keys_output.resize(N);
    d_vals_output.resize(N);
    
    ASSERT_EQUAL(h_keys_output, d_keys_output);
    ASSERT_EQUAL(h_vals_output, d_vals_output);
  }
};
VariableUnitTest<TestReduceByKeyDeviceDevice, IntegralTypes> TestReduceByKeyDeviceDeviceInstance;

