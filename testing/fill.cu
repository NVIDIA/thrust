#include <thrusttest/unittest.h>
#include <thrust/fill.h>


template <class Vector>
void TestFillSimple(void)
{
    typedef typename Vector::value_type T;

    Vector v(5);
    v[0] = 0; v[1] = 1; v[2] = 2; v[3] = 3; v[4] = 4;

    thrust::fill(v.begin() + 1, v.begin() + 4, (T) 7);

    ASSERT_EQUAL(v[0], 0);
    ASSERT_EQUAL(v[1], 7);
    ASSERT_EQUAL(v[2], 7);
    ASSERT_EQUAL(v[3], 7);
    ASSERT_EQUAL(v[4], 4);
    
    thrust::fill(v.begin() + 0, v.begin() + 3, (T) 8);
    
    ASSERT_EQUAL(v[0], 8);
    ASSERT_EQUAL(v[1], 8);
    ASSERT_EQUAL(v[2], 8);
    ASSERT_EQUAL(v[3], 7);
    ASSERT_EQUAL(v[4], 4);
    
    thrust::fill(v.begin() + 2, v.end(), (T) 9);
    
    ASSERT_EQUAL(v[0], 8);
    ASSERT_EQUAL(v[1], 8);
    ASSERT_EQUAL(v[2], 9);
    ASSERT_EQUAL(v[3], 9);
    ASSERT_EQUAL(v[4], 9);

    thrust::fill(v.begin(), v.end(), (T) 1);
    
    ASSERT_EQUAL(v[0], 1);
    ASSERT_EQUAL(v[1], 1);
    ASSERT_EQUAL(v[2], 1);
    ASSERT_EQUAL(v[3], 1);
    ASSERT_EQUAL(v[4], 1);
}
DECLARE_VECTOR_UNITTEST(TestFillSimple);


template <typename T>
void TestFill(size_t n)
{
    thrust::host_vector<T>   h_data = thrusttest::random_integers<T>(n);
    thrust::device_vector<T> d_data = h_data;

    thrust::fill(h_data.begin() + std::min((size_t)1, n), h_data.begin() + std::min((size_t)3, n), (T) 0);
    thrust::fill(d_data.begin() + std::min((size_t)1, n), d_data.begin() + std::min((size_t)3, n), (T) 0);

    ASSERT_EQUAL(h_data, d_data);

    thrust::fill(h_data.begin() + std::min((size_t)117, n), h_data.begin() + std::min((size_t)367, n), (T) 1);
    thrust::fill(d_data.begin() + std::min((size_t)117, n), d_data.begin() + std::min((size_t)367, n), (T) 1);

    ASSERT_EQUAL(h_data, d_data);
    
    thrust::fill(h_data.begin() + std::min((size_t)8, n), h_data.begin() + std::min((size_t)259, n), (T) 2);
    thrust::fill(d_data.begin() + std::min((size_t)8, n), d_data.begin() + std::min((size_t)259, n), (T) 2);

    ASSERT_EQUAL(h_data, d_data);
    
    thrust::fill(h_data.begin() + std::min((size_t)3, n), h_data.end(), (T) 3);
    thrust::fill(d_data.begin() + std::min((size_t)3, n), d_data.end(), (T) 3);

    ASSERT_EQUAL(h_data, d_data);
    
    thrust::fill(h_data.begin(), h_data.end(), (T) 4);
    thrust::fill(d_data.begin(), d_data.end(), (T) 4);

    ASSERT_EQUAL(h_data, d_data);
}
DECLARE_VARIABLE_UNITTEST(TestFill);

