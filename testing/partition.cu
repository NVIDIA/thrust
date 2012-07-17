#include <unittest/unittest.h>
#include <thrust/partition.h>
#include <thrust/count.h>

#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>

template<typename T>
struct is_even
{
    __host__ __device__
    bool operator()(T x) const { return ((int) x % 2) == 0; }
};

typedef unittest::type_list<char,short,int> PartitionTypes;

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
    data[0] =  1; 
    data[1] =  2; 
    data[2] =  1;
    data[3] =  1; 
    data[4] =  2; 

    Vector true_results(2);
    Vector false_results(3);

    thrust::pair<typename Vector::iterator, typename Vector::iterator> ends =
      thrust::stable_partition_copy(data.begin(), data.end(), true_results.begin(), false_results.begin(), is_even<T>());

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
DECLARE_VECTOR_UNITTEST(TestStablePartitionCopySimple);


template <typename T>
struct TestPartition
{
    void operator()(const size_t n)
    {
        // setup ranges
        thrust::host_vector<T>   h_data = unittest::random_integers<T>(n);
        thrust::device_vector<T> d_data = h_data;

        typename thrust::host_vector<T>::iterator   h_iter = thrust::partition(h_data.begin(), h_data.end(), is_even<T>());
        typename thrust::device_vector<T>::iterator d_iter = thrust::partition(d_data.begin(), d_data.end(), is_even<T>());

        thrust::sort(h_data.begin(), h_iter); thrust::sort(h_iter, h_data.end());
        thrust::sort(d_data.begin(), d_iter); thrust::sort(d_iter, d_data.end());

        ASSERT_EQUAL(h_data, d_data);
        ASSERT_EQUAL(h_iter - h_data.begin(), d_iter - d_data.begin());
    }
};
VariableUnitTest<TestPartition, PartitionTypes> TestPartitionInstance;


template <typename T>
struct TestPartitionCopy
{
    void operator()(const size_t n)
    {
        // setup input ranges
        thrust::host_vector<T>   h_data = unittest::random_integers<T>(n);
        thrust::device_vector<T> d_data = h_data;
        
        size_t n_true  = thrust::count_if(h_data.begin(), h_data.end(), is_even<T>());
        size_t n_false = n - n_true;

        // setup output ranges
        thrust::host_vector<T>   h_true_results (n_true,  0);
        thrust::host_vector<T>   h_false_results(n_false, 0);
        thrust::device_vector<T> d_true_results (n_true,  0);
        thrust::device_vector<T> d_false_results(n_false, 0);

        thrust::pair<typename thrust::host_vector<T>::iterator, typename thrust::host_vector<T>::iterator> h_ends
            = thrust::partition_copy(h_data.begin(), h_data.end(), h_true_results.begin(), h_false_results.begin(), is_even<T>());

        thrust::pair<typename thrust::device_vector<T>::iterator, typename thrust::device_vector<T>::iterator> d_ends
            = thrust::partition_copy(d_data.begin(), d_data.end(), d_true_results.begin(), d_false_results.begin(), is_even<T>());

        // check true output
        ASSERT_EQUAL(h_ends.first - h_true_results.begin(), n_true);
        ASSERT_EQUAL(d_ends.first - d_true_results.begin(), n_true);
        thrust::sort(h_true_results.begin(), h_true_results.end());
        thrust::sort(d_true_results.begin(), d_true_results.end());
        ASSERT_EQUAL(h_true_results, d_true_results);

        // check false output
        ASSERT_EQUAL(h_ends.second - h_false_results.begin(), n_false);
        ASSERT_EQUAL(d_ends.second - d_false_results.begin(), n_false);
        thrust::sort(h_false_results.begin(), h_false_results.end());
        thrust::sort(d_false_results.begin(), d_false_results.end());
        ASSERT_EQUAL(h_false_results, d_false_results);
    }
};
VariableUnitTest<TestPartitionCopy, PartitionTypes> TestPartitionCopyInstance;


template <typename T>
struct TestPartitionCopyToDiscardIterator
{
    void operator()(const size_t n)
    {
        // setup input ranges
        thrust::host_vector<T>   h_data = unittest::random_integers<T>(n);
        thrust::device_vector<T> d_data = h_data;
        
        size_t n_true  = thrust::count_if(h_data.begin(), h_data.end(), is_even<T>());
        size_t n_false = n - n_true;

        // mask both ranges
        thrust::pair<thrust::discard_iterator<>, thrust::discard_iterator<> > h_result1 =
            thrust::partition_copy(h_data.begin(),
                                   h_data.end(),
                                   thrust::make_discard_iterator(),
                                   thrust::make_discard_iterator(),
                                   is_even<T>());

        thrust::pair<thrust::discard_iterator<>, thrust::discard_iterator<> > d_result1 =
            thrust::partition_copy(d_data.begin(),
                                   d_data.end(),
                                   thrust::make_discard_iterator(),
                                   thrust::make_discard_iterator(),
                                   is_even<T>());

        thrust::pair<thrust::discard_iterator<>, thrust::discard_iterator<> > reference1 =
            thrust::make_pair(thrust::make_discard_iterator(n_true),
                              thrust::make_discard_iterator(n_false));

        ASSERT_EQUAL_QUIET(reference1, h_result1);
        ASSERT_EQUAL_QUIET(reference1, d_result1);


        // mask the false range
        thrust::host_vector<T> h_trues(n_true);
        thrust::device_vector<T> d_trues(n_true);

        thrust::pair<typename thrust::host_vector<T>::iterator, thrust::discard_iterator<> > h_result2 =
            thrust::partition_copy(h_data.begin(),
                                   h_data.end(),
                                   h_trues.begin(),
                                   thrust::make_discard_iterator(),
                                   is_even<T>());

        thrust::pair<typename thrust::device_vector<T>::iterator, thrust::discard_iterator<> > d_result2 =
            thrust::partition_copy(d_data.begin(),
                                   d_data.end(),
                                   d_trues.begin(),
                                   thrust::make_discard_iterator(),
                                   is_even<T>());

        thrust::pair<typename thrust::host_vector<T>::iterator, thrust::discard_iterator<> > h_reference2 =
            thrust::make_pair(h_trues.begin() + n_true,
                              thrust::make_discard_iterator(n_false));

        thrust::pair<typename thrust::device_vector<T>::iterator, thrust::discard_iterator<> > d_reference2 =
            thrust::make_pair(d_trues.begin() + n_true,
                              thrust::make_discard_iterator(n_false));


        ASSERT_EQUAL(h_trues, d_trues);
        ASSERT_EQUAL_QUIET(h_reference2, h_result2);
        ASSERT_EQUAL_QUIET(d_reference2, d_result2);



        // mask the true range
        thrust::host_vector<T> h_falses(n_false);
        thrust::device_vector<T> d_falses(n_false);

        thrust::pair<thrust::discard_iterator<>, typename thrust::host_vector<T>::iterator> h_result3 =
            thrust::partition_copy(h_data.begin(),
                                   h_data.end(),
                                   thrust::make_discard_iterator(),
                                   h_falses.begin(),
                                   is_even<T>());

        thrust::pair<thrust::discard_iterator<>, typename thrust::device_vector<T>::iterator> d_result3 =
            thrust::partition_copy(d_data.begin(),
                                   d_data.end(),
                                   thrust::make_discard_iterator(),
                                   d_falses.begin(),
                                   is_even<T>());

        thrust::pair<thrust::discard_iterator<>, typename thrust::host_vector<T>::iterator> h_reference3 =
            thrust::make_pair(thrust::make_discard_iterator(n_true),
                              h_falses.begin() + n_false);

        thrust::pair<thrust::discard_iterator<>, typename thrust::device_vector<T>::iterator> d_reference3 =
            thrust::make_pair(thrust::make_discard_iterator(n_true),
                              d_falses.begin() + n_false);


        ASSERT_EQUAL(h_falses, d_falses);
        ASSERT_EQUAL_QUIET(h_reference3, h_result3);
        ASSERT_EQUAL_QUIET(d_reference3, d_result3);
    }
};
VariableUnitTest<TestPartitionCopyToDiscardIterator, PartitionTypes> TestPartitionCopyToDiscardIteratorInstance;


template <typename T>
struct TestStablePartition
{
    void operator()(const size_t n)
    {
        // setup ranges
        thrust::host_vector<T>   h_data = unittest::random_integers<T>(n);
        thrust::device_vector<T> d_data = h_data;

        typename thrust::host_vector<T>::iterator   h_iter = thrust::stable_partition(h_data.begin(), h_data.end(), is_even<T>());
        typename thrust::device_vector<T>::iterator d_iter = thrust::stable_partition(d_data.begin(), d_data.end(), is_even<T>());

        ASSERT_EQUAL(h_data, d_data);
        ASSERT_EQUAL(h_iter - h_data.begin(), d_iter - d_data.begin());
    }
};
VariableUnitTest<TestStablePartition, PartitionTypes> TestStablePartitionInstance;


template <typename T>
struct TestStablePartitionCopy
{
    void operator()(const size_t n)
    {
        // setup input ranges
        thrust::host_vector<T>   h_data = unittest::random_integers<T>(n);
        thrust::device_vector<T> d_data = h_data;
        
        size_t n_true  = thrust::count_if(h_data.begin(), h_data.end(), is_even<T>());
        size_t n_false = n - n_true;

        // setup output ranges
        thrust::host_vector<T>   h_true_results (n_true,  0);
        thrust::host_vector<T>   h_false_results(n_false, 0);
        thrust::device_vector<T> d_true_results (n_true,  0);
        thrust::device_vector<T> d_false_results(n_false, 0);

        thrust::pair<typename thrust::host_vector<T>::iterator, typename thrust::host_vector<T>::iterator> h_ends
            = thrust::stable_partition_copy(h_data.begin(), h_data.end(), h_true_results.begin(), h_false_results.begin(), is_even<T>());

        thrust::pair<typename thrust::device_vector<T>::iterator, typename thrust::device_vector<T>::iterator> d_ends
            = thrust::stable_partition_copy(d_data.begin(), d_data.end(), d_true_results.begin(), d_false_results.begin(), is_even<T>());

        // check true output
        ASSERT_EQUAL(h_ends.first - h_true_results.begin(), n_true);
        ASSERT_EQUAL(d_ends.first - d_true_results.begin(), n_true);
        ASSERT_EQUAL(h_true_results, d_true_results);

        // check false output
        ASSERT_EQUAL(h_ends.second - h_false_results.begin(), n_false);
        ASSERT_EQUAL(d_ends.second - d_false_results.begin(), n_false);
        ASSERT_EQUAL(h_false_results, d_false_results);
    }
};
VariableUnitTest<TestStablePartitionCopy, PartitionTypes> TestStablePartitionCopyInstance;


template <typename T>
struct TestStablePartitionCopyToDiscardIterator
{
    void operator()(const size_t n)
    {
        // setup input ranges
        thrust::host_vector<T>   h_data = unittest::random_integers<T>(n);
        thrust::device_vector<T> d_data = h_data;
        
        size_t n_true  = thrust::count_if(h_data.begin(), h_data.end(), is_even<T>());
        size_t n_false = n - n_true;

        // mask both ranges
        thrust::pair<thrust::discard_iterator<>, thrust::discard_iterator<> > h_result1 =
            thrust::stable_partition_copy(h_data.begin(),
                                          h_data.end(),
                                          thrust::make_discard_iterator(),
                                          thrust::make_discard_iterator(),
                                          is_even<T>());

        thrust::pair<thrust::discard_iterator<>, thrust::discard_iterator<> > d_result1 =
            thrust::stable_partition_copy(d_data.begin(),
                                          d_data.end(),
                                          thrust::make_discard_iterator(),
                                          thrust::make_discard_iterator(),
                                          is_even<T>());

        thrust::pair<thrust::discard_iterator<>, thrust::discard_iterator<> > reference1 =
            thrust::make_pair(thrust::make_discard_iterator(n_true),
                              thrust::make_discard_iterator(n_false));

        ASSERT_EQUAL_QUIET(reference1, h_result1);
        ASSERT_EQUAL_QUIET(reference1, d_result1);


        // mask the false range
        thrust::host_vector<T> h_trues(n_true);
        thrust::device_vector<T> d_trues(n_true);

        thrust::pair<typename thrust::host_vector<T>::iterator, thrust::discard_iterator<> > h_result2 =
            thrust::stable_partition_copy(h_data.begin(),
                                          h_data.end(),
                                          h_trues.begin(),
                                          thrust::make_discard_iterator(),
                                          is_even<T>());

        thrust::pair<typename thrust::device_vector<T>::iterator, thrust::discard_iterator<> > d_result2 =
            thrust::stable_partition_copy(d_data.begin(),
                                          d_data.end(),
                                          d_trues.begin(),
                                          thrust::make_discard_iterator(),
                                          is_even<T>());

        thrust::pair<typename thrust::host_vector<T>::iterator, thrust::discard_iterator<> > h_reference2 =
            thrust::make_pair(h_trues.begin() + n_true,
                              thrust::make_discard_iterator(n_false));

        thrust::pair<typename thrust::device_vector<T>::iterator, thrust::discard_iterator<> > d_reference2 =
            thrust::make_pair(d_trues.begin() + n_true,
                              thrust::make_discard_iterator(n_false));


        ASSERT_EQUAL(h_trues, d_trues);
        ASSERT_EQUAL_QUIET(h_reference2, h_result2);
        ASSERT_EQUAL_QUIET(d_reference2, d_result2);



        // mask the true range
        thrust::host_vector<T> h_falses(n_false);
        thrust::device_vector<T> d_falses(n_false);

        thrust::pair<thrust::discard_iterator<>, typename thrust::host_vector<T>::iterator> h_result3 =
            thrust::stable_partition_copy(h_data.begin(),
                                          h_data.end(),
                                          thrust::make_discard_iterator(),
                                          h_falses.begin(),
                                          is_even<T>());

        thrust::pair<thrust::discard_iterator<>, typename thrust::device_vector<T>::iterator> d_result3 =
            thrust::stable_partition_copy(d_data.begin(),
                                          d_data.end(),
                                          thrust::make_discard_iterator(),
                                          d_falses.begin(),
                                          is_even<T>());

        thrust::pair<thrust::discard_iterator<>, typename thrust::host_vector<T>::iterator> h_reference3 =
            thrust::make_pair(thrust::make_discard_iterator(n_true),
                              h_falses.begin() + n_false);

        thrust::pair<thrust::discard_iterator<>, typename thrust::device_vector<T>::iterator> d_reference3 =
            thrust::make_pair(thrust::make_discard_iterator(n_true),
                              d_falses.begin() + n_false);


        ASSERT_EQUAL(h_falses, d_falses);
        ASSERT_EQUAL_QUIET(h_reference3, h_result3);
        ASSERT_EQUAL_QUIET(d_reference3, d_result3);
    }
};
VariableUnitTest<TestStablePartitionCopyToDiscardIterator, PartitionTypes> TestStablePartitionCopyToDiscardIteratorInstance;


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


struct my_system : thrust::device_system<my_system> {};

template<typename ForwardIterator,
         typename Predicate>
ForwardIterator partition(my_system,
                          ForwardIterator first,
                          ForwardIterator last,
                          Predicate pred)
{
    *first = 13;
    return first;
}

void TestPartitionDispatchExplicit()
{
    thrust::device_vector<int> vec(1);

    my_system sys;
    thrust::partition(sys,
                      vec.begin(),
                      vec.begin(),
                      0);

    ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestPartitionDispatchExplicit);

void TestPartitionDispatchImplicit()
{
    thrust::device_vector<int> vec(1);

    thrust::partition(thrust::retag<my_system>(vec.begin()),
                      thrust::retag<my_system>(vec.begin()),
                      0);

    ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestPartitionDispatchImplicit);

template<typename InputIterator,
         typename OutputIterator1,
         typename OutputIterator2,
         typename Predicate>
  thrust::pair<OutputIterator1,OutputIterator2>
    partition_copy(my_system,
                   InputIterator first,
                   InputIterator last,
                   OutputIterator1 out_true,
                   OutputIterator2 out_false,
                   Predicate pred)
{
  *first = 13;
  return thrust::make_pair(out_true,out_false);
}

void TestPartitionCopyDispatchExplicit()
{
    thrust::device_vector<int> vec(1);

    my_system sys;
    thrust::partition_copy(sys,
                           vec.begin(),
                           vec.begin(),
                           vec.begin(),
                           vec.begin(),
                           0);

    ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestPartitionCopyDispatchExplicit);

void TestPartitionCopyDispatchImplicit()
{
    thrust::device_vector<int> vec(1);

    thrust::partition_copy(thrust::retag<my_system>(vec.begin()),
                           thrust::retag<my_system>(vec.begin()),
                           thrust::retag<my_system>(vec.begin()),
                           thrust::retag<my_system>(vec.begin()),
                           0);

    ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestPartitionCopyDispatchImplicit);

template<typename ForwardIterator,
         typename Predicate>
ForwardIterator stable_partition(my_system,
                                 ForwardIterator first,
                                 ForwardIterator last,
                                 Predicate pred)
{
    *first = 13;
    return first;
}

void TestStablePartitionDispatchExplicit()
{
    thrust::device_vector<int> vec(1);

    my_system sys;
    thrust::stable_partition(sys,
                             vec.begin(),
                             vec.begin(),
                             0);

    ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestStablePartitionDispatchExplicit);

void TestStablePartitionDispatchImplicit()
{
    thrust::device_vector<int> vec(1);

    thrust::stable_partition(thrust::retag<my_system>(vec.begin()),
                             thrust::retag<my_system>(vec.begin()),
                             0);

    ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestStablePartitionDispatchImplicit);

template<typename InputIterator,
         typename OutputIterator1,
         typename OutputIterator2,
         typename Predicate>
  thrust::pair<OutputIterator1,OutputIterator2>
    stable_partition_copy(my_system,
                          InputIterator first,
                          InputIterator last,
                          OutputIterator1 out_true,
                          OutputIterator2 out_false,
                          Predicate pred)
{
  *first = 13;
  return thrust::make_pair(out_true,out_false);
}

void TestStablePartitionCopyDispatchExplicit()
{
    thrust::device_vector<int> vec(1);

    my_system sys;
    thrust::stable_partition_copy(sys,
                                  vec.begin(),
                                  vec.begin(),
                                  vec.begin(),
                                  vec.begin(),
                                  0);

    ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestStablePartitionCopyDispatchExplicit);

void TestStablePartitionCopyDispatchImplicit()
{
    thrust::device_vector<int> vec(1);

    thrust::stable_partition_copy(thrust::retag<my_system>(vec.begin()),
                                  thrust::retag<my_system>(vec.begin()),
                                  thrust::retag<my_system>(vec.begin()),
                                  thrust::retag<my_system>(vec.begin()),
                                  0);

    ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestStablePartitionCopyDispatchImplicit);

