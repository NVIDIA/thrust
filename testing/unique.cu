#include <thrusttest/unittest.h>
#include <thrust/unique.h>
#include <thrust/functional.h>

template<typename T>
struct is_equal_div_10_unique
{
    __host__ __device__
    bool operator()(const T x, const T& y) const { return ((int) x / 10) == ((int) y / 10); }
};

template<typename T>
struct is_equal_div_2_unique
{
    __host__ __device__
    bool operator()(const T x, const T& y) const { return ((int) x / 2) == ((int) y / 2); }
};



template<typename Vector>
void TestUniqueSimple(void)
{
    typedef typename Vector::value_type T;

    Vector data(10);
    data[0] = 11; 
    data[1] = 11; 
    data[2] = 12;
    data[3] = 20; 
    data[4] = 29; 
    data[5] = 21; 
    data[6] = 21; 
    data[7] = 31; 
    data[8] = 31; 
    data[9] = 37; 

    typename Vector::iterator new_last;
    
    new_last = thrust::unique(data.begin(), data.end());

    ASSERT_EQUAL(new_last - data.begin(), 7);
    ASSERT_EQUAL(data[0], 11);
    ASSERT_EQUAL(data[1], 12);
    ASSERT_EQUAL(data[2], 20);
    ASSERT_EQUAL(data[3], 29);
    ASSERT_EQUAL(data[4], 21);
    ASSERT_EQUAL(data[5], 31);
    ASSERT_EQUAL(data[6], 37);

    new_last = thrust::unique(data.begin(), new_last, is_equal_div_10_unique<T>());

    ASSERT_EQUAL(new_last - data.begin(), 3);
    ASSERT_EQUAL(data[0], 11);
    ASSERT_EQUAL(data[1], 20);
    ASSERT_EQUAL(data[2], 31);
}
DECLARE_VECTOR_UNITTEST(TestUniqueSimple);


template<typename T>
void TestUnique(const size_t n)
{
    thrust::host_vector<T>   h_data = thrusttest::random_samples<T>(n);

    // round floats to ints so some duplicates occur
    for(size_t i = 0; i < n; i++)
        h_data[i] = T(int(h_data[i]));

    thrust::device_vector<T> d_data = h_data;

    typename thrust::host_vector<T>::iterator   h_new_last;
    typename thrust::device_vector<T>::iterator d_new_last;

    // using operator==
    h_new_last = thrust::unique(h_data.begin(), h_data.end());
    d_new_last = thrust::unique(d_data.begin(), d_data.end());

    ASSERT_EQUAL(h_new_last - h_data.begin(), d_new_last - d_data.begin());
   
    h_data.resize(h_new_last - h_data.begin());
    d_data.resize(d_new_last - d_data.begin());

    // using custom binary predicate
    h_new_last = thrust::unique(h_data.begin(), h_data.end(), is_equal_div_2_unique<T>());
    d_new_last = thrust::unique(d_data.begin(), d_data.end(), is_equal_div_2_unique<T>());

    ASSERT_EQUAL(h_new_last - h_data.begin(), d_new_last - d_data.begin());
}
DECLARE_VARIABLE_UNITTEST(TestUnique);
