#include <thrusttest/unittest.h>
#include <thrust/copy.h>

#include <list>
#include <iterator>

void TestCopyFromConstIterator(void)
{
    typedef int T;

    std::vector<T> v(5);
    v[0] = 0; v[1] = 1; v[2] = 2; v[3] = 3; v[4] = 4;

    std::vector<int>::const_iterator begin = v.begin();
    std::vector<int>::const_iterator end = v.end();

    // copy to host_vector
    thrust::host_vector<T> h(5, (T) 10);
    thrust::host_vector<T>::iterator h_result = thrust::copy(begin, end, h.begin());
    ASSERT_EQUAL(h[0], 0);
    ASSERT_EQUAL(h[1], 1);
    ASSERT_EQUAL(h[2], 2);
    ASSERT_EQUAL(h[3], 3);
    ASSERT_EQUAL(h[4], 4);
    ASSERT_EQUAL_QUIET(h_result, h.end());

    // copy to device_vector
    thrust::device_vector<T> d(5, (T) 10);
    thrust::device_vector<T>::iterator d_result = thrust::copy(begin, end, d.begin());
    ASSERT_EQUAL(d[0], 0);
    ASSERT_EQUAL(d[1], 1);
    ASSERT_EQUAL(d[2], 2);
    ASSERT_EQUAL(d[3], 3);
    ASSERT_EQUAL(d[4], 4);
    ASSERT_EQUAL_QUIET(d_result, d.end());
}
DECLARE_UNITTEST(TestCopyFromConstIterator);

template <class Vector>
void TestCopyMatchingTypes(void)
{
    typedef typename Vector::value_type T;

    Vector v(5);
    v[0] = 0; v[1] = 1; v[2] = 2; v[3] = 3; v[4] = 4;

    // copy to host_vector
    thrust::host_vector<T> h(5, (T) 10);
    typename thrust::host_vector<T>::iterator h_result = thrust::copy(v.begin(), v.end(), h.begin());
    ASSERT_EQUAL(h[0], 0);
    ASSERT_EQUAL(h[1], 1);
    ASSERT_EQUAL(h[2], 2);
    ASSERT_EQUAL(h[3], 3);
    ASSERT_EQUAL(h[4], 4);
    ASSERT_EQUAL_QUIET(h_result, h.end());

    // copy to device_vector
    thrust::device_vector<T> d(5, (T) 10);
    typename thrust::device_vector<T>::iterator d_result = thrust::copy(v.begin(), v.end(), d.begin());
    ASSERT_EQUAL(d[0], 0);
    ASSERT_EQUAL(d[1], 1);
    ASSERT_EQUAL(d[2], 2);
    ASSERT_EQUAL(d[3], 3);
    ASSERT_EQUAL(d[4], 4);
    ASSERT_EQUAL_QUIET(d_result, d.end());
}
DECLARE_VECTOR_UNITTEST(TestCopyMatchingTypes);

template <class Vector>
void TestCopyMixedTypes(void)
{
    typedef typename Vector::value_type T;

    Vector v(5);
    v[0] = 0; v[1] = 1; v[2] = 2; v[3] = 3; v[4] = 4;

    // copy to host_vector with different type
    thrust::host_vector<float> h(5, (float) 10);
    typename thrust::host_vector<float>::iterator h_result = thrust::copy(v.begin(), v.end(), h.begin());

    ASSERT_EQUAL(h[0], 0);
    ASSERT_EQUAL(h[1], 1);
    ASSERT_EQUAL(h[2], 2);
    ASSERT_EQUAL(h[3], 3);
    ASSERT_EQUAL(h[4], 4);
    ASSERT_EQUAL_QUIET(h_result, h.end());

    // copy to device_vector with different type
    thrust::device_vector<float> d(5, (float) 10);
    typename thrust::device_vector<float>::iterator d_result = thrust::copy(v.begin(), v.end(), d.begin());
    ASSERT_EQUAL(d[0], 0);
    ASSERT_EQUAL(d[1], 1);
    ASSERT_EQUAL(d[2], 2);
    ASSERT_EQUAL(d[3], 3);
    ASSERT_EQUAL(d[4], 4);
    ASSERT_EQUAL_QUIET(d_result, d.end());
}
DECLARE_VECTOR_UNITTEST(TestCopyMixedTypes);


void TestCopyVectorBool(void)
{
    std::vector<bool> v(3);
    v[0] = true; v[1] = false; v[2] = true;

    thrust::host_vector<bool> h(3);
    thrust::device_vector<bool> d(3);
    
    thrust::copy(v.begin(), v.end(), h.begin());
    thrust::copy(v.begin(), v.end(), d.begin());

    ASSERT_EQUAL(h[0], true);
    ASSERT_EQUAL(h[1], false);
    ASSERT_EQUAL(h[2], true);

    ASSERT_EQUAL(d[0], true);
    ASSERT_EQUAL(d[1], false);
    ASSERT_EQUAL(d[2], true);
}
DECLARE_UNITTEST(TestCopyVectorBool);


template <class Vector>
void TestCopyListTo(void)
{
    typedef typename Vector::value_type T;

    // copy from list to Vector
    std::list<T> l;
    l.push_back(0);
    l.push_back(1);
    l.push_back(2);
    l.push_back(3);
    l.push_back(4);
   
    Vector v(l.size());

    typename Vector::iterator v_result = thrust::copy(l.begin(), l.end(), v.begin());

    ASSERT_EQUAL(v[0], 0);
    ASSERT_EQUAL(v[1], 1);
    ASSERT_EQUAL(v[2], 2);
    ASSERT_EQUAL(v[3], 3);
    ASSERT_EQUAL(v[4], 4);
    ASSERT_EQUAL_QUIET(v_result, v.end());

    l.clear();

    std::back_insert_iterator< std::list<T> > l_result = thrust::copy(v.begin(), v.end(), std::back_insert_iterator< std::list<T> >(l));

    ASSERT_EQUAL(l.size(), 5);

    typename std::list<T>::const_iterator iter = l.begin();
    ASSERT_EQUAL(*iter, 0);  iter++;
    ASSERT_EQUAL(*iter, 1);  iter++;
    ASSERT_EQUAL(*iter, 2);  iter++;
    ASSERT_EQUAL(*iter, 3);  iter++;
    ASSERT_EQUAL(*iter, 4);  iter++;
}
DECLARE_VECTOR_UNITTEST(TestCopyListTo);


template <class Vector>
void TestCopyWhenSimple(void)
{
    typedef typename Vector::value_type T;

    Vector v(5);
    v[0] = 0; v[1] = 1; v[2] = 2; v[3] = 3; v[4] = 4;

    Vector stencil(5, (T) 0);
    stencil[1] = 1; stencil[3] = 1; stencil[4] = 1;

    Vector dest(5, (T) 10);

    typename Vector::iterator dest_result = thrust::copy_when(v.begin(), v.end(), stencil.begin(), dest.begin());
    ASSERT_EQUAL(dest[0], 10);
    ASSERT_EQUAL(dest[1],  1);
    ASSERT_EQUAL(dest[2], 10);
    ASSERT_EQUAL(dest[3],  3);
    ASSERT_EQUAL(dest[4],  4);
    ASSERT_EQUAL_QUIET(dest_result, dest.end());
}
DECLARE_VECTOR_UNITTEST(TestCopyWhenSimple);


template<typename T>
struct is_even
{
    __host__ __device__
    bool operator()(T x) { return (x & 1) == 0; }
};

template<typename T>
struct is_true
{
    __host__ __device__
    bool operator()(T x) { return x ? true : false; }
};
    

template <class Vector>
void TestCopyIfSimple(void)
{
    typedef typename Vector::value_type T;

    Vector v(5);
    v[0] = 0; v[1] = 1; v[2] = 2; v[3] = 3; v[4] = 4;

    Vector dest(3);

    typename Vector::iterator dest_end = thrust::copy_if(v.begin(), v.end(), dest.begin(), is_even<T>());

    ASSERT_EQUAL(0, dest[0]);
    ASSERT_EQUAL(2, dest[1]);
    ASSERT_EQUAL(4, dest[2]);
    ASSERT_EQUAL_QUIET(dest.end(), dest_end);
}
DECLARE_VECTOR_UNITTEST(TestCopyIfSimple);


template <typename T>
void TestCopyIf(const size_t n)
{
    thrust::host_vector<T> h_data = thrusttest::random_samples<T>(n);
    thrust::device_vector<T> d_data = h_data;

    thrust::host_vector<T> h_result(n);
    thrust::device_vector<T> d_result(n);

    is_true<T> pred;

    typename thrust::host_vector<T>::iterator h_new_end = thrust::copy_if(h_data.begin(), h_data.end(), h_result.begin(), pred);
    typename thrust::device_vector<T>::iterator d_new_end = thrust::copy_if(d_data.begin(), d_data.end(), d_result.begin(), pred);

    h_result.resize(h_new_end - h_result.begin());
    d_result.resize(d_new_end - d_result.begin());

    ASSERT_EQUAL(h_result, d_result);
}
DECLARE_VARIABLE_UNITTEST(TestCopyIf);


template <class Vector>
void TestCopyIfStencilSimple(void)
{
    typedef typename Vector::value_type T;

    Vector v(5);
    v[0] = 0; v[1] = 1; v[2] = 2; v[3] = 3; v[4] = 4;

    Vector s(5);
    s[0] = 1; s[1] = 1; s[2] = 0; s[3] = 1; s[4] = 0;

    Vector dest(3);

    typename Vector::iterator dest_end = thrust::copy_if(v.begin(), v.end(), s.begin(), dest.begin(), is_true<T>());

    ASSERT_EQUAL(0, dest[0]);
    ASSERT_EQUAL(1, dest[1]);
    ASSERT_EQUAL(3, dest[2]);
    ASSERT_EQUAL_QUIET(dest.end(), dest_end);
}
DECLARE_VECTOR_UNITTEST(TestCopyIfStencilSimple);


template <typename T>
void TestCopyIfStencil(const size_t n)
{
    thrust::host_vector<T> h_data = thrusttest::random_samples<T>(n);
    thrust::device_vector<T> d_data = h_data;

    thrust::host_vector<T> h_stencil = thrusttest::random_samples<T>(n);
    thrust::device_vector<T> d_stencil = thrusttest::random_samples<T>(n);

    thrust::host_vector<T> h_result(n);
    thrust::device_vector<T> d_result(n);

    is_true<T> pred;

    typename thrust::host_vector<T>::iterator h_new_end = thrust::copy_if(h_data.begin(), h_data.end(), h_stencil.begin(), h_result.begin(), pred);
    typename thrust::device_vector<T>::iterator d_new_end = thrust::copy_if(d_data.begin(), d_data.end(), d_stencil.begin(), d_result.begin(), pred);

    h_result.resize(h_new_end - h_result.begin());
    d_result.resize(d_new_end - d_result.begin());

    ASSERT_EQUAL(h_result, d_result);
}
DECLARE_VARIABLE_UNITTEST(TestCopyIfStencil);

