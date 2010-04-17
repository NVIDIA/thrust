#include <unittest/unittest.h>
#include <thrust/partition.h>

#include <thrust/sort.h>

template<typename T>
struct is_even
{
    __host__ __device__
    bool operator()(T x) const { return ((int) x % 2) == 0; }
};


template<typename Vector>
void TestPartitionSimple(void)
{
    typedef typename Vector::value_type T;

    Vector data(5);
    data[0] = 1; 
    data[1] = 2; 
    data[2] = 1;
    data[3] = 1; 
    data[4] = 2; 

    typename Vector::iterator iter = thrust::partition(data.begin(), data.end(), is_even<T>());

    Vector ref(5);
    ref[0] = 2;
    ref[1] = 2;
    ref[2] = 1;
    ref[3] = 1;
    ref[4] = 1;

    ASSERT_EQUAL( int(iter - data.begin()), 2 );
    ASSERT_EQUAL(data, ref);
}
DECLARE_VECTOR_UNITTEST(TestPartitionSimple);


template<typename Vector>
void TestPartitionCopySimple(void)
{
    typedef typename Vector::value_type T;

    Vector data(5);
    data[0] =  1; 
    data[1] =  2; 
    data[2] =  1;
    data[3] =  1; 
    data[4] =  2; 

    Vector result(5);

    thrust::experimental::partition_copy(data.begin(), data.end(), result.begin(), is_even<T>());

    Vector ref(5);
    ref[0] =  2;
    ref[1] =  2;
    ref[2] =  1;
    ref[3] =  1;
    ref[4] =  1;

    ASSERT_EQUAL(result, ref);
}
DECLARE_VECTOR_UNITTEST(TestPartitionCopySimple);


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
void TestPartition(const size_t n)
{
    thrust::host_vector<T>   h_data = unittest::random_integers<T>(n);
    thrust::device_vector<T> d_data = h_data;

    typename thrust::host_vector<T>::iterator   h_iter = thrust::partition(h_data.begin(), h_data.end(), is_even<T>());
    typename thrust::device_vector<T>::iterator d_iter = thrust::partition(d_data.begin(), d_data.end(), is_even<T>());

    thrust::sort(h_data.begin(), h_iter); thrust::sort(h_iter, h_data.end());
    thrust::sort(d_data.begin(), d_iter); thrust::sort(d_iter, d_data.end());

    ASSERT_EQUAL(h_iter - h_data.begin(), d_iter - d_data.begin());
    ASSERT_EQUAL(h_data, d_data);
}
DECLARE_VARIABLE_UNITTEST(TestPartition);


template <typename T>
void TestPartitionCopy(const size_t n)
{
    thrust::host_vector<T>   h_data = unittest::random_integers<T>(n);
    thrust::device_vector<T> d_data = h_data;

    thrust::host_vector<T>   h_result(n);
    thrust::device_vector<T> d_result(n);
    
    typename thrust::host_vector<T>::iterator   h_iter = thrust::experimental::partition_copy(h_data.begin(), h_data.end(), h_result.begin(), is_even<T>());
    typename thrust::device_vector<T>::iterator d_iter = thrust::experimental::partition_copy(d_data.begin(), d_data.end(), d_result.begin(), is_even<T>());

    thrust::sort(h_result.begin(), h_iter); thrust::sort(h_iter, h_result.end());
    thrust::sort(d_result.begin(), d_iter); thrust::sort(d_iter, d_result.end());

    ASSERT_EQUAL(h_iter - h_result.begin(), d_iter - d_result.begin());
    ASSERT_EQUAL(h_result, d_result);
}
DECLARE_VARIABLE_UNITTEST(TestPartitionCopy);


template <typename T>
void TestStablePartition(const size_t n)
{
    thrust::host_vector<T>   h_data = unittest::random_integers<T>(n);
    thrust::device_vector<T> d_data = h_data;

    typename thrust::host_vector<T>::iterator   h_iter = thrust::stable_partition(h_data.begin(), h_data.end(), is_even<T>());
    typename thrust::device_vector<T>::iterator d_iter = thrust::stable_partition(d_data.begin(), d_data.end(), is_even<T>());

    ASSERT_EQUAL(h_iter - h_data.begin(), d_iter - d_data.begin());
    ASSERT_EQUAL(h_data, d_data);
}
DECLARE_VARIABLE_UNITTEST(TestStablePartition);


template <typename T>
void TestStablePartitionCopy(const size_t n)
{
    thrust::host_vector<T>   h_data = unittest::random_integers<T>(n);
    thrust::device_vector<T> d_data = h_data;

    thrust::host_vector<T>   h_result(n);
    thrust::device_vector<T> d_result(n);
    
    typename thrust::host_vector<T>::iterator   h_iter = thrust::experimental::stable_partition_copy(h_data.begin(), h_data.end(), h_result.begin(), is_even<T>());
    typename thrust::device_vector<T>::iterator d_iter = thrust::experimental::stable_partition_copy(d_data.begin(), d_data.end(), d_result.begin(), is_even<T>());

    ASSERT_EQUAL(h_iter - h_result.begin(), d_iter - d_result.begin());
    ASSERT_EQUAL(h_result, d_result);
}
DECLARE_VARIABLE_UNITTEST(TestStablePartitionCopy);

