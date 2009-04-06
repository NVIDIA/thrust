#include <komradetest/unittest.h>
#include <komrade/extrema.h>

template <class Vector>
void TestMinElementSimple(void)
{
    typedef typename Vector::value_type T;

    Vector data(6);
    data[0] = 3;
    data[1] = 5;
    data[2] = 1;
    data[3] = 2;
    data[4] = 5;
    data[5] = 1;

    ASSERT_EQUAL( *komrade::min_element(data.begin(), data.end()), 1);
    ASSERT_EQUAL( komrade::min_element(data.begin(), data.end()) - data.begin(), 2);
    
    ASSERT_EQUAL( *komrade::min_element(data.begin(), data.end(), komrade::greater<T>()), 5);
    ASSERT_EQUAL( komrade::min_element(data.begin(), data.end(), komrade::greater<T>()) - data.begin(), 1);
}
DECLARE_VECTOR_UNITTEST(TestMinElementSimple);

template<typename T>
void TestMinElement(const size_t n)
{
    komrade::host_vector<T> h_data = komradetest::random_samples<T>(n);
    komrade::device_vector<T> d_data = h_data;

    typename komrade::host_vector<T>::iterator   h_min = komrade::min_element(h_data.begin(), h_data.end());
    typename komrade::device_vector<T>::iterator d_min = komrade::min_element(d_data.begin(), d_data.end());

    ASSERT_EQUAL(h_min - h_data.begin(), d_min - d_data.begin());
    
    typename komrade::host_vector<T>::iterator   h_max = komrade::min_element(h_data.begin(), h_data.end(), komrade::greater<T>());
    typename komrade::device_vector<T>::iterator d_max = komrade::min_element(d_data.begin(), d_data.end(), komrade::greater<T>());

    ASSERT_EQUAL(h_max - h_data.begin(), d_max - d_data.begin());
}
DECLARE_VARIABLE_UNITTEST(TestMinElement);

