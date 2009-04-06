#include <komradetest/unittest.h>
#include <komrade/partition.h>

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

    komrade::stable_partition_copy(data.begin(), data.end(), result.begin(), is_even<T>());

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
    komrade::host_vector<T> h_data = komradetest::random_samples<T>(n);
    komrade::device_vector<T> d_data = h_data;

    komrade::host_vector<T> h_result(n);
    komrade::device_vector<T> d_result(n);

    komrade::stable_partition_copy(h_data.begin(), h_data.end(), h_result.begin(), is_true<T>());
    komrade::stable_partition_copy(d_data.begin(), d_data.end(), d_result.begin(), is_true<T>());

    ASSERT_EQUAL(h_result, d_result);
    
    komrade::stable_partition_copy(h_data.begin(), h_data.end(), h_result.begin(), is_even<T>());
    komrade::stable_partition_copy(d_data.begin(), d_data.end(), d_result.begin(), is_even<T>());

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

    komrade::stable_partition(data.begin(), data.end(), is_even<T>());

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
    komrade::host_vector<T> h_data = komradetest::random_samples<T>(n);
    komrade::device_vector<T> d_data = h_data;

    komrade::stable_partition(h_data.begin(), h_data.end(), is_true<T>());
    komrade::stable_partition(d_data.begin(), d_data.end(), is_true<T>());

    ASSERT_EQUAL(h_data, d_data);
    
    komrade::stable_partition(h_data.begin(), h_data.end(), is_even<T>());
    komrade::stable_partition(d_data.begin(), d_data.end(), is_even<T>());

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

    komrade::partition_copy(data.begin(), data.end(), result.begin(), is_even<T>());

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
    komrade::host_vector<T> h_data = komradetest::random_samples<T>(n);
    komrade::device_vector<T> d_data = h_data;

    komrade::host_vector<T> h_result(n);
    komrade::device_vector<T> d_result(n);

    komrade::partition_copy(h_data.begin(), h_data.end(), h_result.begin(), is_true<T>());
    komrade::partition_copy(d_data.begin(), d_data.end(), d_result.begin(), is_true<T>());

    ASSERT_EQUAL(h_result, d_result);
    
    komrade::partition_copy(h_data.begin(), h_data.end(), h_result.begin(), is_even<T>());
    komrade::partition_copy(d_data.begin(), d_data.end(), d_result.begin(), is_even<T>());

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

    komrade::partition(data.begin(), data.end(), is_even<T>());

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
    komrade::host_vector<T> h_data = komradetest::random_samples<T>(n);
    komrade::device_vector<T> d_data = h_data;

    komrade::partition(h_data.begin(), h_data.end(), is_true<T>());
    komrade::partition(d_data.begin(), d_data.end(), is_true<T>());

    ASSERT_EQUAL(h_data, d_data);
    
    komrade::partition(h_data.begin(), h_data.end(), is_even<T>());
    komrade::partition(d_data.begin(), d_data.end(), is_even<T>());

    ASSERT_EQUAL(h_data, d_data);
}
DECLARE_VARIABLE_UNITTEST(TestPartition);

