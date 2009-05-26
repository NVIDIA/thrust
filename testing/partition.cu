#include <thrusttest/unittest.h>
#include <thrust/partition.h>

template<typename T>
struct is_even
{
    __host__ __device__
    bool operator()(T x) const { return ((int) x & 1) == 0; }
};

template<typename T>
struct is_true
{
    __host__ __device__
    bool operator()(const T x) { return x ? true : false; }
};


template<typename Vector>
void TestStablePartitionCopySimple(void)
{
    typedef typename Vector::value_type T;

    Vector data(5);
    data[0] = 1; 
    data[1] = 2; 
    data[2] = 1;
    data[3] = 3; 
    data[4] = 2; 

    Vector result(5);

    thrust::experimental::stable_partition_copy(data.begin(), data.end(), result.begin(), is_even<T>());

    Vector ref(5);
    ref[0] = 2;
    ref[1] = 2;
    ref[2] = 1;
    ref[3] = 1;
    ref[4] = 3;

    ASSERT_EQUAL(result, ref);
}
DECLARE_VECTOR_UNITTEST(TestStablePartitionCopySimple);


template <typename T>
void TestStablePartitionCopy(const size_t n)
{
    thrust::host_vector<T> h_data = thrusttest::random_samples<T>(n);
    thrust::device_vector<T> d_data = h_data;

    thrust::host_vector<T> h_result(n);
    thrust::device_vector<T> d_result(n);

    thrust::experimental::stable_partition_copy(h_data.begin(), h_data.end(), h_result.begin(), is_true<T>());
    thrust::experimental::stable_partition_copy(d_data.begin(), d_data.end(), d_result.begin(), is_true<T>());

    ASSERT_EQUAL(h_result, d_result);
    
    thrust::experimental::stable_partition_copy(h_data.begin(), h_data.end(), h_result.begin(), is_even<T>());
    thrust::experimental::stable_partition_copy(d_data.begin(), d_data.end(), d_result.begin(), is_even<T>());

    ASSERT_EQUAL(h_result, d_result);
}
DECLARE_VARIABLE_UNITTEST(TestStablePartitionCopy);


template<typename Vector>
void TestStablePartitionSimple(void)
{
    typedef typename Vector::value_type T;

    Vector data(5);
    data[0] =  1; 
    data[1] =  2; 
    data[2] =  1;
    data[3] =  3; 
    data[4] =  2; 

    thrust::stable_partition(data.begin(), data.end(), is_even<T>());

    Vector ref(5);
    ref[0] =  2;
    ref[1] =  2;
    ref[2] =  1;
    ref[3] =  1;
    ref[4] =  3;

    ASSERT_EQUAL(data, ref);
}
DECLARE_VECTOR_UNITTEST(TestStablePartitionSimple);


template <typename T>
void TestStablePartition(const size_t n)
{
    thrust::host_vector<T> h_data = thrusttest::random_samples<T>(n);
    thrust::device_vector<T> d_data = h_data;

    thrust::stable_partition(h_data.begin(), h_data.end(), is_true<T>());
    thrust::stable_partition(d_data.begin(), d_data.end(), is_true<T>());

    ASSERT_EQUAL(h_data, d_data);
    
    thrust::stable_partition(h_data.begin(), h_data.end(), is_even<T>());
    thrust::stable_partition(d_data.begin(), d_data.end(), is_even<T>());

    ASSERT_EQUAL(h_data, d_data);
}
DECLARE_VARIABLE_UNITTEST(TestStablePartition);

template<typename Vector>
void TestPartitionCopySimple(void)
{
    typedef typename Vector::value_type T;

    Vector data(5);
    data[0] =  1; 
    data[1] =  2; 
    data[2] =  1;
    data[3] =  3; 
    data[4] =  2; 

    Vector result(5);

    thrust::experimental::partition_copy(data.begin(), data.end(), result.begin(), is_even<T>());

    Vector ref(5);
    ref[0] =  2;
    ref[1] =  2;
    ref[2] =  1;
    ref[3] =  1;
    ref[4] =  3;

    ASSERT_EQUAL(result, ref);
}
DECLARE_VECTOR_UNITTEST(TestPartitionCopySimple);


template <typename T>
void TestPartitionCopy(const size_t n)
{
    thrust::host_vector<T> h_data = thrusttest::random_samples<T>(n);
    thrust::device_vector<T> d_data = h_data;

    thrust::host_vector<T> h_result(n);
    thrust::device_vector<T> d_result(n);

    thrust::experimental::partition_copy(h_data.begin(), h_data.end(), h_result.begin(), is_true<T>());
    thrust::experimental::partition_copy(d_data.begin(), d_data.end(), d_result.begin(), is_true<T>());

    ASSERT_EQUAL(h_result, d_result);
    
    thrust::experimental::partition_copy(h_data.begin(), h_data.end(), h_result.begin(), is_even<T>());
    thrust::experimental::partition_copy(d_data.begin(), d_data.end(), d_result.begin(), is_even<T>());

    ASSERT_EQUAL(h_result, d_result);
}
DECLARE_VARIABLE_UNITTEST(TestPartitionCopy);


template<typename Vector>
void TestPartitionSimple(void)
{
    typedef typename Vector::value_type T;

    Vector data(5);
    data[0] = 1; 
    data[1] = 2; 
    data[2] = 1;
    data[3] = 3; 
    data[4] = 2; 

    thrust::partition(data.begin(), data.end(), is_even<T>());

    Vector ref(5);
    ref[0] = 2;
    ref[1] = 2;
    ref[2] = 1;
    ref[3] = 1;
    ref[4] = 3;

    ASSERT_EQUAL(data, ref);
}
DECLARE_VECTOR_UNITTEST(TestPartitionSimple);


template <typename T>
void TestPartition(const size_t n)
{
    thrust::host_vector<T> h_data = thrusttest::random_samples<T>(n);
    thrust::device_vector<T> d_data = h_data;

    thrust::partition(h_data.begin(), h_data.end(), is_true<T>());
    thrust::partition(d_data.begin(), d_data.end(), is_true<T>());

    ASSERT_EQUAL(h_data, d_data);
    
    thrust::partition(h_data.begin(), h_data.end(), is_even<T>());
    thrust::partition(d_data.begin(), d_data.end(), is_even<T>());

    ASSERT_EQUAL(h_data, d_data);
}
DECLARE_VARIABLE_UNITTEST(TestPartition);

