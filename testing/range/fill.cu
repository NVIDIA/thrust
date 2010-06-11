#include <unittest/unittest.h>
#include <thrust/range/algorithm/fill.h>
#include <thrust/pair.h>


template <class Vector>
void TestRangeFillSimple(void)
{
    typedef typename Vector::value_type T;

    Vector v(5);
    v[0] = 0; v[1] = 1; v[2] = 2; v[3] = 3; v[4] = 4;

    using namespace thrust::experimental::range;

    fill(thrust::make_pair(v.begin() + 1, v.begin() + 4), (T) 7);

    ASSERT_EQUAL(0, v[0]);
    ASSERT_EQUAL(7, v[1]);
    ASSERT_EQUAL(7, v[2]);
    ASSERT_EQUAL(7, v[3]);
    ASSERT_EQUAL(4, v[4]);
    
    fill(thrust::make_pair(v.begin() + 0, v.begin() + 3), (T) 8);
    
    ASSERT_EQUAL(8, v[0]);
    ASSERT_EQUAL(8, v[1]);
    ASSERT_EQUAL(8, v[2]);
    ASSERT_EQUAL(7, v[3]);
    ASSERT_EQUAL(4, v[4]);
    
    fill(thrust::make_pair(v.begin() + 2, v.end()), (T) 9);
    
    ASSERT_EQUAL(8, v[0]);
    ASSERT_EQUAL(8, v[1]);
    ASSERT_EQUAL(9, v[2]);
    ASSERT_EQUAL(9, v[3]);
    ASSERT_EQUAL(9, v[4]);

    fill(thrust::make_pair(v.begin(), v.end()), (T) 1);
    
    ASSERT_EQUAL(1, v[0]);
    ASSERT_EQUAL(1, v[1]);
    ASSERT_EQUAL(1, v[2]);
    ASSERT_EQUAL(1, v[3]);
    ASSERT_EQUAL(1, v[4]);
}
DECLARE_VECTOR_UNITTEST(TestRangeFillSimple);


template <class Vector>
void TestRangeFillMixedTypes(void)
{
    typedef typename Vector::value_type T;

    Vector v(4);

    using namespace thrust::experimental::range;

    fill(v, (long) 10);
    
    ASSERT_EQUAL(10, v[0]);
    ASSERT_EQUAL(10, v[1]);
    ASSERT_EQUAL(10, v[2]);
    ASSERT_EQUAL(10, v[3]);
    
    fill(v, (float) 20);
    
    ASSERT_EQUAL(20, v[0]);
    ASSERT_EQUAL(20, v[1]);
    ASSERT_EQUAL(20, v[2]);
    ASSERT_EQUAL(20, v[3]);
}
DECLARE_VECTOR_UNITTEST(TestRangeFillMixedTypes);


template <typename T>
void TestRangeFill(size_t n)
{
    using namespace thrust::experimental::range;

    thrust::host_vector<T>   h_data = unittest::random_integers<T>(n);
    thrust::device_vector<T> d_data = h_data;

    fill(thrust::make_pair(h_data.begin() + std::min((size_t)1, n), h_data.begin() + std::min((size_t)3, n)), (T) 0);
    fill(thrust::make_pair(d_data.begin() + std::min((size_t)1, n), d_data.begin() + std::min((size_t)3, n)), (T) 0);

    ASSERT_EQUAL(h_data, d_data);

    fill(thrust::make_pair(h_data.begin() + std::min((size_t)117, n), h_data.begin() + std::min((size_t)367, n)), (T) 1);
    fill(thrust::make_pair(d_data.begin() + std::min((size_t)117, n), d_data.begin() + std::min((size_t)367, n)), (T) 1);

    ASSERT_EQUAL(h_data, d_data);
    
    fill(thrust::make_pair(h_data.begin() + std::min((size_t)8, n), h_data.begin() + std::min((size_t)259, n)), (T) 2);
    fill(thrust::make_pair(d_data.begin() + std::min((size_t)8, n), d_data.begin() + std::min((size_t)259, n)), (T) 2);

    ASSERT_EQUAL(h_data, d_data);
    
    fill(thrust::make_pair(h_data.begin() + std::min((size_t)3, n), h_data.end()), (T) 3);
    fill(thrust::make_pair(d_data.begin() + std::min((size_t)3, n), d_data.end()), (T) 3);

    ASSERT_EQUAL(h_data, d_data);
    
    fill(h_data, (T) 4);
    fill(d_data, (T) 4);

    ASSERT_EQUAL(h_data, d_data);
}
DECLARE_VARIABLE_UNITTEST(TestRangeFill);

