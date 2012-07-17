#include <unittest/unittest.h>
#include <thrust/extrema.h>

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

    ASSERT_EQUAL( *thrust::min_element(data.begin(), data.end()), 1);
    ASSERT_EQUAL( thrust::min_element(data.begin(), data.end()) - data.begin(), 2);
    
    ASSERT_EQUAL( *thrust::min_element(data.begin(), data.end(), thrust::greater<T>()), 5);
    ASSERT_EQUAL( thrust::min_element(data.begin(), data.end(), thrust::greater<T>()) - data.begin(), 1);
}
DECLARE_VECTOR_UNITTEST(TestMinElementSimple);

template<typename T>
void TestMinElement(const size_t n)
{
    thrust::host_vector<T> h_data = unittest::random_samples<T>(n);
    thrust::device_vector<T> d_data = h_data;

    typename thrust::host_vector<T>::iterator   h_min = thrust::min_element(h_data.begin(), h_data.end());
    typename thrust::device_vector<T>::iterator d_min = thrust::min_element(d_data.begin(), d_data.end());

    ASSERT_EQUAL(h_min - h_data.begin(), d_min - d_data.begin());
    
    typename thrust::host_vector<T>::iterator   h_max = thrust::min_element(h_data.begin(), h_data.end(), thrust::greater<T>());
    typename thrust::device_vector<T>::iterator d_max = thrust::min_element(d_data.begin(), d_data.end(), thrust::greater<T>());

    ASSERT_EQUAL(h_max - h_data.begin(), d_max - d_data.begin());
}
DECLARE_VARIABLE_UNITTEST(TestMinElement);

struct my_system : thrust::device_system<my_system> {};

template<typename ForwardIterator>
ForwardIterator min_element(my_system, ForwardIterator first, ForwardIterator)
{
    *first = 13;
    return first;
}

void TestMinElementDispatchExplicit()
{
    thrust::device_vector<int> vec(1);

    my_system sys;
    thrust::min_element(sys, vec.begin(), vec.end());

    ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestMinElementDispatchExplicit);


void TestMinElementDispatchImplicit()
{
    thrust::device_vector<int> vec(1);

    thrust::min_element(thrust::retag<my_system>(vec.begin()),
                        thrust::retag<my_system>(vec.end()));

    ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestMinElementDispatchImplicit);

