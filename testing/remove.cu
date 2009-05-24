#include <thrusttest/unittest.h>
#include <thrust/remove.h>
#include <stdexcept>

template<typename T>
struct is_even_remove
{
    __host__ __device__
    bool operator()(T x) { return (x & 1) == 0; }
};

template<typename T>
struct is_true_remove
{
    __host__ __device__
    bool operator()(T x) { return x ? true : false; }
};

template<typename Vector>
void TestRemoveCopyIfSimple(void)
{
    typedef typename Vector::value_type T;

    Vector data(5);
    data[0] =  1; 
    data[1] =  2; 
    data[2] =  1;
    data[3] =  3; 
    data[4] =  2; 

    Vector result(5);

    typename Vector::iterator end = thrust::remove_copy_if(data.begin(), 
                                                            data.end(), 
                                                            result.begin(), 
                                                            is_even_remove<T>());

    ASSERT_EQUAL(end - result.begin(), 3);

    ASSERT_EQUAL(result[0], 1);
    ASSERT_EQUAL(result[1], 1);
    ASSERT_EQUAL(result[2], 3);
}
DECLARE_VECTOR_UNITTEST(TestRemoveCopyIfSimple);


template<typename T>
void TestRemoveCopyIf(const size_t n)
{
    thrust::host_vector<T> h_data = thrusttest::random_samples<T>(n);
    thrust::device_vector<T> d_data = h_data;

    thrust::host_vector<T> h_result(n);
    thrust::device_vector<T> d_result(n);

    is_true_remove<T> pred;

    typename thrust::host_vector<T>::iterator h_new_end = thrust::remove_copy_if(h_data.begin(), h_data.end(), h_result.begin(), pred);
    typename thrust::device_vector<T>::iterator d_new_end = thrust::remove_copy_if(d_data.begin(), d_data.end(), d_result.begin(), pred);

    h_result.resize(h_new_end - h_result.begin());
    d_result.resize(d_new_end - d_result.begin());

    ASSERT_EQUAL(h_result, d_result);
}
DECLARE_VARIABLE_UNITTEST(TestRemoveCopyIf);


template<typename Vector>
void TestRemoveCopySimple(void)
{
    typedef typename Vector::value_type T;

    Vector data(5);
    data[0] =  1; 
    data[1] =  2; 
    data[2] =  1;
    data[3] =  3; 
    data[4] =  2; 

    Vector result(5);

    typename Vector::iterator end = thrust::remove_copy(data.begin(), 
                                                         data.end(), 
                                                         result.begin(), 
                                                         (T) 2);

    ASSERT_EQUAL(end - result.begin(), 3);

    ASSERT_EQUAL(result[0], 1);
    ASSERT_EQUAL(result[1], 1);
    ASSERT_EQUAL(result[2], 3);
}
DECLARE_VECTOR_UNITTEST(TestRemoveCopySimple);


template<typename T>
void TestRemoveCopy(const size_t n)
{
    thrust::host_vector<T> h_data = thrusttest::random_samples<T>(n);
    thrust::device_vector<T> d_data = h_data;

    thrust::host_vector<T> h_result(n);
    thrust::device_vector<T> d_result(n);

    T remove_me = 0;
    if(n > 0) remove_me = h_data[0];

    typename thrust::host_vector<T>::iterator h_new_end = thrust::remove_copy(h_data.begin(), h_data.end(), h_result.begin(), remove_me);
    typename thrust::device_vector<T>::iterator d_new_end = thrust::remove_copy(d_data.begin(), d_data.end(), d_result.begin(), remove_me);

    h_result.resize(h_new_end - h_result.begin());
    d_result.resize(d_new_end - d_result.begin());

    ASSERT_EQUAL(h_result, d_result);
}
DECLARE_VARIABLE_UNITTEST(TestRemoveCopy);


template<typename Vector>
void TestRemoveIfSimple(void)
{
    typedef typename Vector::value_type T;

    Vector data(5);
    data[0] =  1; 
    data[1] =  2; 
    data[2] =  1;
    data[3] =  3; 
    data[4] =  2; 

    typename Vector::iterator end = thrust::remove_if(data.begin(), 
                                                       data.end(), 
                                                       is_even_remove<T>());

    ASSERT_EQUAL(end - data.begin(), 3);

    ASSERT_EQUAL(data[0], 1);
    ASSERT_EQUAL(data[1], 1);
    ASSERT_EQUAL(data[2], 3);
}
DECLARE_VECTOR_UNITTEST(TestRemoveIfSimple);


template<typename T>
void TestRemoveIf(const size_t n)
{
    thrust::host_vector<T> h_data = thrusttest::random_samples<T>(n);
    thrust::device_vector<T> d_data = h_data;

    is_true_remove<T> pred;

    typename thrust::host_vector<T>::iterator h_new_end = thrust::remove_if(h_data.begin(), h_data.end(), pred);
    typename thrust::device_vector<T>::iterator d_new_end = thrust::remove_if(d_data.begin(), d_data.end(), pred);

    h_data.resize(h_new_end - h_data.begin());
    d_data.resize(d_new_end - d_data.begin());

    ASSERT_EQUAL(h_data, d_data);
}
DECLARE_VARIABLE_UNITTEST(TestRemoveIf);


template<typename Vector>
void TestRemoveSimple(void)
{
    typedef typename Vector::value_type T;

    Vector data(5);
    data[0] =  1; 
    data[1] =  2; 
    data[2] =  1;
    data[3] =  3; 
    data[4] =  2; 

    typename Vector::iterator end = thrust::remove(data.begin(), 
                                                    data.end(), 
                                                    (T) 2);

    ASSERT_EQUAL(end - data.begin(), 3);

    ASSERT_EQUAL(data[0], 1);
    ASSERT_EQUAL(data[1], 1);
    ASSERT_EQUAL(data[2], 3);
}
DECLARE_VECTOR_UNITTEST(TestRemoveSimple);


template<typename T>
void TestRemove(const size_t n)
{
    thrust::host_vector<T> h_data = thrusttest::random_samples<T>(n);
    thrust::device_vector<T> d_data = h_data;

    T remove_me = 0;
    if(n > 0) remove_me = h_data[0];

    typename thrust::host_vector<T>::iterator h_new_end = thrust::remove(h_data.begin(), h_data.end(), remove_me);
    typename thrust::device_vector<T>::iterator d_new_end = thrust::remove(d_data.begin(), d_data.end(), remove_me);

    h_data.resize(h_new_end - h_data.begin());
    d_data.resize(d_new_end - d_data.begin());

    ASSERT_EQUAL(h_data, d_data);
}
DECLARE_VARIABLE_UNITTEST(TestRemove);

