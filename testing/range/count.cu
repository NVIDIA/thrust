#include <unittest/unittest.h>
#include <thrust/range/algorithm/count.h>

template <class Vector>
void TestRangeCountSimple(void)
{
    typedef typename Vector::value_type T;

    Vector data(5);
    data[0] = 1; data[1] = 1; data[2] = 0; data[3] = 0; data[4] = 1;

    using namespace thrust::experimental::range;

    ASSERT_EQUAL(2, count(data, 0));
    ASSERT_EQUAL(3, count(data, 1));
    ASSERT_EQUAL(0, count(data, 2));
}
DECLARE_VECTOR_UNITTEST(TestRangeCountSimple);

template <typename T>
void TestRangeCount(const size_t n)
{
    thrust::host_vector<T>   h_data = unittest::random_samples<T>(n);
    thrust::device_vector<T> d_data = h_data;

    using namespace thrust::experimental::range;

    size_t cpu_result = count(h_data, T(5));
    size_t gpu_result = count(d_data, T(5));

    ASSERT_EQUAL(cpu_result, gpu_result);
}
DECLARE_VARIABLE_UNITTEST(TestRangeCount);




template <typename T>
struct greater_than_five
{
  __host__ __device__ bool operator()(const T &x) const {return x > 5;}
};

template <class Vector>
void TestRangeCountIfSimple(void)
{
    typedef typename Vector::value_type T;

    Vector data(5);
    data[0] = 1; data[1] = 6; data[2] = 1; data[3] = 9; data[4] = 2;

    using namespace thrust::experimental::range;

    ASSERT_EQUAL(2, count_if(data, greater_than_five<T>()));
}
DECLARE_VECTOR_UNITTEST(TestRangeCountIfSimple);


template <typename T>
void TestRangeCountIf(const size_t n)
{
    thrust::host_vector<T>   h_data = unittest::random_samples<T>(n);
    thrust::device_vector<T> d_data = h_data;

    using namespace thrust::experimental::range;

    size_t cpu_result = count_if(h_data, greater_than_five<T>());
    size_t gpu_result = count_if(d_data, greater_than_five<T>());

    ASSERT_EQUAL(cpu_result, gpu_result);
}
DECLARE_VARIABLE_UNITTEST(TestRangeCountIf);

