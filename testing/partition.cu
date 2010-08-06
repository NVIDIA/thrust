#include <unittest/unittest.h>
#include <thrust/partition.h>
#include <thrust/count.h>

#include <thrust/iterator/zip_iterator.h>
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
    typedef typename Vector::iterator   Iterator;

    Vector data(5);
    data[0] = 1; 
    data[1] = 2; 
    data[2] = 1;
    data[3] = 1; 
    data[4] = 2; 

    Iterator iter = thrust::partition(data.begin(), data.end(), is_even<T>());

    Vector ref(5);
    ref[0] = 2;
    ref[1] = 2;
    ref[2] = 1;
    ref[3] = 1;
    ref[4] = 1;

    ASSERT_EQUAL(iter - data.begin(), 2);
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

    Vector true_results(2);
    Vector false_results(3);

    thrust::pair<typename Vector::iterator, typename Vector::iterator> ends =
      thrust::partition_copy(data.begin(), data.end(), true_results.begin(), false_results.begin(), is_even<T>());

    Vector true_ref(2);
    true_ref[0] =  2;
    true_ref[1] =  2;

    Vector false_ref(3);
    false_ref[0] =  1;
    false_ref[1] =  1;
    false_ref[2] =  1;

    ASSERT_EQUAL(2, ends.first - true_results.begin());
    ASSERT_EQUAL(3, ends.second - false_results.begin());
    ASSERT_EQUAL(true_ref, true_results);
    ASSERT_EQUAL(false_ref, false_results);
}
DECLARE_VECTOR_UNITTEST(TestPartitionCopySimple);


template<typename Vector>
void TestStablePartitionSimple(void)
{
    typedef typename Vector::value_type T;
    typedef typename Vector::iterator   Iterator;

    Vector data(5);
    data[0] =  1; 
    data[1] =  2; 
    data[2] =  1;
    data[3] =  3; 
    data[4] =  2; 

    Iterator iter = thrust::stable_partition(data.begin(), data.end(), is_even<T>());

    Vector ref(5);
    ref[0] =  2;
    ref[1] =  2;
    ref[2] =  1;
    ref[3] =  1;
    ref[4] =  3;

    ASSERT_EQUAL(iter - data.begin(), 2);
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

    thrust::host_vector<T>   h_true_results(thrust::count_if(h_data.begin(), h_data.end(), is_even<T>()));
    thrust::host_vector<T>   h_false_results(n - h_true_results.size());
    thrust::device_vector<T> d_true_results(h_true_results.size());
    thrust::device_vector<T> d_false_results(h_false_results.size());
    
    thrust::pair<typename thrust::host_vector<T>::iterator, typename thrust::host_vector<T>::iterator> h_ends
      = thrust::partition_copy(h_data.begin(), h_data.end(), h_true_results.begin(), h_false_results.begin(), is_even<T>());

    thrust::pair<typename thrust::device_vector<T>::iterator, typename thrust::device_vector<T>::iterator> d_ends
      = thrust::partition_copy(d_data.begin(), d_data.end(), d_true_results.begin(), d_false_results.begin(), is_even<T>());

    thrust::sort(h_true_results.begin(), h_ends.first);
    thrust::sort(h_false_results.begin(), h_ends.second);

    thrust::sort(d_true_results.begin(), d_ends.first);
    thrust::sort(d_false_results.begin(), d_ends.second);

    ASSERT_EQUAL(h_ends.first - h_true_results.begin(), d_ends.first - d_true_results.begin());
    ASSERT_EQUAL(h_ends.second - h_false_results.begin(), d_ends.second - d_false_results.begin());

    ASSERT_EQUAL(h_true_results, d_true_results);
    ASSERT_EQUAL(h_false_results, d_false_results);
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


struct is_ordered
{
    template <typename Tuple>
    __host__ __device__
    bool operator()(const Tuple& t) const
    {
        return thrust::get<0>(t) <= thrust::get<1>(t);
    }
};


template<typename Vector>
void TestPartitionZipIterator(void)
{
    typedef typename Vector::value_type T;

    Vector data1(5);
    Vector data2(5);

    data1[0] = 1;  data2[0] = 2; 
    data1[1] = 2;  data2[1] = 1;
    data1[2] = 1;  data2[2] = 2;
    data1[3] = 1;  data2[3] = 2;
    data1[4] = 2;  data2[4] = 1;

    typedef typename Vector::iterator           Iterator;
    typedef thrust::tuple<Iterator,Iterator>    IteratorTuple;
    typedef thrust::zip_iterator<IteratorTuple> ZipIterator;

    ZipIterator begin = thrust::make_zip_iterator(thrust::make_tuple(data1.begin(), data2.begin()));
    ZipIterator end   = thrust::make_zip_iterator(thrust::make_tuple(data1.end(),   data2.end()));

    ZipIterator iter = thrust::partition(begin, end, is_ordered());

    Vector ref1(5);
    Vector ref2(5);

    ref1[0] = 1; ref2[0] = 2;
    ref1[1] = 1; ref2[1] = 2;
    ref1[2] = 1; ref2[2] = 2;
    ref1[3] = 2; ref2[3] = 1;
    ref1[4] = 2; ref2[4] = 1;

    ASSERT_EQUAL(iter - begin, 3);
    ASSERT_EQUAL(data1, ref1);
    ASSERT_EQUAL(data2, ref2);
}
DECLARE_VECTOR_UNITTEST(TestPartitionZipIterator);


template<typename Vector>
void TestStablePartitionZipIterator(void)
{
    typedef typename Vector::value_type T;

    Vector data1(5);
    Vector data2(5);

    data1[0] = 1;  data2[0] = 2; 
    data1[1] = 2;  data2[1] = 0;
    data1[2] = 1;  data2[2] = 3;
    data1[3] = 1;  data2[3] = 2;
    data1[4] = 2;  data2[4] = 1;

    typedef typename Vector::iterator           Iterator;
    typedef thrust::tuple<Iterator,Iterator>    IteratorTuple;
    typedef thrust::zip_iterator<IteratorTuple> ZipIterator;

    ZipIterator begin = thrust::make_zip_iterator(thrust::make_tuple(data1.begin(), data2.begin()));
    ZipIterator end   = thrust::make_zip_iterator(thrust::make_tuple(data1.end(),   data2.end()));

    ZipIterator iter = thrust::stable_partition(begin, end, is_ordered());

    Vector ref1(5);
    Vector ref2(5);

    ref1[0] = 1; ref2[0] = 2;
    ref1[1] = 1; ref2[1] = 3;
    ref1[2] = 1; ref2[2] = 2;
    ref1[3] = 2; ref2[3] = 0;
    ref1[4] = 2; ref2[4] = 1;

    ASSERT_EQUAL(data1, ref1);
    ASSERT_EQUAL(data2, ref2);
    ASSERT_EQUAL(iter - begin, 3);
}
DECLARE_VECTOR_UNITTEST(TestStablePartitionZipIterator);

