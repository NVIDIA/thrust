#include <unittest/unittest.h>
#include <thrust/range/algorithm/transform.h>
#include <thrust/range/algorithm/reduce.h>
#include <thrust/range/algorithm/sequence.h>

template <class Vector>
void TestRangeTransformReduceSimple(void)
{
    typedef typename Vector::value_type T;

    Vector data(3);
    data[0] = 1; data[1] = -2; data[2] = 3;

    using namespace thrust::experimental::range;

    T init = 10;
    T result = reduce(transform(data, thrust::negate<T>()), init, thrust::plus<T>());

    ASSERT_EQUAL(8, result);
}
DECLARE_VECTOR_UNITTEST(TestRangeTransformReduceSimple);

template <typename T>
void TestRangeTransformReduce(const size_t n)
{
    thrust::host_vector<T>   h_data = unittest::random_integers<T>(n);
    thrust::device_vector<T> d_data = h_data;

    T init = 13;

    using namespace thrust::experimental::range;

    T cpu_result = reduce(transform(h_data, thrust::negate<T>()), init, thrust::plus<T>());
    T gpu_result = reduce(transform(d_data, thrust::negate<T>()), init, thrust::plus<T>());

    ASSERT_ALMOST_EQUAL(cpu_result, gpu_result);
}
DECLARE_VARIABLE_UNITTEST(TestRangeTransformReduce);

template <typename T>
void TestRangeTransformReduceFromConst(const size_t n)
{
    thrust::host_vector<T>   h_data = unittest::random_integers<T>(n);
    thrust::device_vector<T> d_data = h_data;

    T init = 13;

    using namespace thrust::experimental::range;

    T cpu_result = reduce(transform(static_cast<const thrust::host_vector<T>&>(h_data), thrust::negate<T>()), init, thrust::plus<T>());
    T gpu_result = reduce(transform(static_cast<const thrust::device_vector<T>&>(d_data), thrust::negate<T>()), init, thrust::plus<T>());

    ASSERT_ALMOST_EQUAL(cpu_result, gpu_result);
}
DECLARE_VARIABLE_UNITTEST(TestRangeTransformReduceFromConst);

template <class Vector>
void TestRangeTransformReduceSequence(void)
{
    typedef typename Vector::value_type T;
    typedef typename thrust::iterator_space<typename Vector::iterator>::type space;

    thrust::counting_iterator<T, space> first(1);

    using namespace thrust::experimental::range;

    T result = reduce(transform(sequence<space>(1, 4), thrust::negate<short>()), 0, thrust::plus<short>());

    ASSERT_EQUAL(result, -6);
}
DECLARE_VECTOR_UNITTEST(TestRangeTransformReduceSequence);

