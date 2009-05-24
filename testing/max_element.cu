#include <thrusttest/unittest.h>
#include <thrust/extrema.h>

template <class Vector>
void TestMaxElementSimple(void)
{
    typedef typename Vector::value_type T;

    Vector data(6);
    data[0] = 3;
    data[1] = 5;
    data[2] = 1;
    data[3] = 2;
    data[4] = 5;
    data[5] = 1;

    ASSERT_EQUAL( *thrust::max_element(data.begin(), data.end()), 5);
    ASSERT_EQUAL( thrust::max_element(data.begin(), data.end()) - data.begin(), 1);
    
    ASSERT_EQUAL( *thrust::max_element(data.begin(), data.end(), thrust::greater<T>()), 1);
    ASSERT_EQUAL( thrust::max_element(data.begin(), data.end(), thrust::greater<T>()) - data.begin(), 2);
}
DECLARE_VECTOR_UNITTEST(TestMaxElementSimple);

template<typename T>
void TestMaxElement(const size_t n)
{
    thrust::host_vector<T> h_data = thrusttest::random_samples<T>(n);
    thrust::device_vector<T> d_data = h_data;

    typename thrust::host_vector<T>::iterator   h_max = thrust::max_element(h_data.begin(), h_data.end());
    typename thrust::device_vector<T>::iterator d_max = thrust::max_element(d_data.begin(), d_data.end());

    ASSERT_EQUAL(h_max - h_data.begin(), d_max - d_data.begin());
    
    typename thrust::host_vector<T>::iterator   h_min = thrust::max_element(h_data.begin(), h_data.end(), thrust::greater<T>());
    typename thrust::device_vector<T>::iterator d_min = thrust::max_element(d_data.begin(), d_data.end(), thrust::greater<T>());

    ASSERT_EQUAL(h_min - h_data.begin(), d_min - d_data.begin());
}
DECLARE_VARIABLE_UNITTEST(TestMaxElement);

