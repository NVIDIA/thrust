#include <thrusttest/unittest.h>
#include <thrust/count.h>

template <class Vector>
void TestCountSimple(void)
{
    typedef typename Vector::value_type T;

    Vector data(5);
    data[0] = 1; data[1] = 1; data[2] = 0; data[3] = 0; data[4] = 1;

    ASSERT_EQUAL(thrust::count(data.begin(), data.end(), 0), 2);
    ASSERT_EQUAL(thrust::count(data.begin(), data.end(), 1), 3);
    ASSERT_EQUAL(thrust::count(data.begin(), data.end(), 2), 0);
}
DECLARE_VECTOR_UNITTEST(TestCountSimple);

template <typename T>
void TestCount(const size_t n)
{
    thrust::host_vector<T>   h_data = thrusttest::random_samples<T>(n);
    thrust::device_vector<T> d_data = h_data;

    size_t cpu_result = thrust::count(h_data.begin(), h_data.end(), 5);
    size_t gpu_result = thrust::count(d_data.begin(), d_data.end(), 5);

    ASSERT_EQUAL(cpu_result, gpu_result);
}
DECLARE_VARIABLE_UNITTEST(TestCount);




template <typename T>
struct greater_than_five
{
  __host__ __device__ bool operator()(const T &x) const {return x > 5;}
};

template <class Vector>
void TestCountIfSimple(void)
{
    typedef typename Vector::value_type T;

    Vector data(5);
    data[0] = 1; data[1] = 6; data[2] = 1; data[3] = 9; data[4] = 2;

    ASSERT_EQUAL(thrust::count_if(data.begin(), data.end(), greater_than_five<T>()), 2);
}
DECLARE_VECTOR_UNITTEST(TestCountIfSimple);


template <typename T>
void TestCountIf(const size_t n)
{
    thrust::host_vector<T>   h_data = thrusttest::random_samples<T>(n);
    thrust::device_vector<T> d_data = h_data;

    size_t cpu_result = thrust::count_if(h_data.begin(), h_data.end(), greater_than_five<T>());
    size_t gpu_result = thrust::count_if(d_data.begin(), d_data.end(), greater_than_five<T>());

    ASSERT_EQUAL(cpu_result, gpu_result);
}
DECLARE_VARIABLE_UNITTEST(TestCountIf);

