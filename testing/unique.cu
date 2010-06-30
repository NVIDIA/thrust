#include <unittest/unittest.h>
#include <thrust/unique.h>
#include <thrust/functional.h>

template<typename T>
struct is_equal_div_10_unique
{
    __host__ __device__
    bool operator()(const T x, const T& y) const { return ((int) x / 10) == ((int) y / 10); }
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
struct TestUnique
{
    void operator()(const size_t n)
    {
        thrust::host_vector<T>   h_data = unittest::random_integers<bool>(n);
        thrust::device_vector<T> d_data = h_data;

        typename thrust::host_vector<T>::iterator   h_new_last;
        typename thrust::device_vector<T>::iterator d_new_last;

        h_new_last = thrust::unique(h_data.begin(), h_data.end());
        d_new_last = thrust::unique(d_data.begin(), d_data.end());

        ASSERT_EQUAL(h_new_last - h_data.begin(), d_new_last - d_data.begin());
        
        h_data.resize(h_new_last - h_data.begin());
        d_data.resize(d_new_last - d_data.begin());

        ASSERT_EQUAL(h_data, d_data);
    }
};
VariableUnitTest<TestUnique, IntegralTypes> TestUniqueInstance;


template<typename Vector>
void TestUniqueCopySimple(void)
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
    
    Vector output(10, -1);

    typename Vector::iterator new_last;
    
    new_last = thrust::unique_copy(data.begin(), data.end(), output.begin());

    ASSERT_EQUAL(new_last - output.begin(), 7);
    ASSERT_EQUAL(output[0], 11);
    ASSERT_EQUAL(output[1], 12);
    ASSERT_EQUAL(output[2], 20);
    ASSERT_EQUAL(output[3], 29);
    ASSERT_EQUAL(output[4], 21);
    ASSERT_EQUAL(output[5], 31);
    ASSERT_EQUAL(output[6], 37);

    new_last = thrust::unique_copy(output.begin(), new_last, data.begin(), is_equal_div_10_unique<T>());

    ASSERT_EQUAL(new_last - data.begin(), 3);
    ASSERT_EQUAL(data[0], 11);
    ASSERT_EQUAL(data[1], 20);
    ASSERT_EQUAL(data[2], 31);
}
DECLARE_VECTOR_UNITTEST(TestUniqueCopySimple);


template<typename T>
struct TestUniqueCopy
{
    void operator()(const size_t n)
    {
        thrust::host_vector<T>   h_data = unittest::random_integers<bool>(n);
        thrust::device_vector<T> d_data = h_data;
        
        thrust::host_vector<T>   h_output(n);
        thrust::device_vector<T> d_output(n);

        typename thrust::host_vector<T>::iterator   h_new_last;
        typename thrust::device_vector<T>::iterator d_new_last;

        h_new_last = thrust::unique_copy(h_data.begin(), h_data.end(), h_output.begin());
        d_new_last = thrust::unique_copy(d_data.begin(), d_data.end(), d_output.begin());

        ASSERT_EQUAL(h_new_last - h_output.begin(), d_new_last - d_output.begin());
        
        h_data.resize(h_new_last - h_output.begin());
        d_data.resize(d_new_last - d_output.begin());

        ASSERT_EQUAL(h_output, d_output);
    }
};
VariableUnitTest<TestUniqueCopy, IntegralTypes> TestUniqueCopyInstance;


template <typename Vector>
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

template <typename Vector>
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

