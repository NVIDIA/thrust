#include <komradetest/unittest.h>
#include <komrade/extrema.h>

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

    ASSERT_EQUAL( *komrade::max_element(data.begin(), data.end()), 5);
    ASSERT_EQUAL( komrade::max_element(data.begin(), data.end()) - data.begin(), 1);
    
    ASSERT_EQUAL( *komrade::max_element(data.begin(), data.end(), komrade::greater<T>()), 1);
    ASSERT_EQUAL( komrade::max_element(data.begin(), data.end(), komrade::greater<T>()) - data.begin(), 2);
}
DECLARE_VECTOR_UNITTEST(TestMaxElementSimple);

template<typename T>
void TestMaxElement(const size_t n)
{
    komrade::host_vector<T> h_data = komradetest::random_samples<T>(n);
    komrade::device_vector<T> d_data = h_data;

    typename komrade::host_vector<T>::iterator   h_max = komrade::max_element(h_data.begin(), h_data.end());
    typename komrade::device_vector<T>::iterator d_max = komrade::max_element(d_data.begin(), d_data.end());

    ASSERT_EQUAL(h_max - h_data.begin(), d_max - d_data.begin());
    
    typename komrade::host_vector<T>::iterator   h_min = komrade::max_element(h_data.begin(), h_data.end(), komrade::greater<T>());
    typename komrade::device_vector<T>::iterator d_min = komrade::max_element(d_data.begin(), d_data.end(), komrade::greater<T>());

    ASSERT_EQUAL(h_min - h_data.begin(), d_min - d_data.begin());
}
DECLARE_VARIABLE_UNITTEST(TestMaxElement);

