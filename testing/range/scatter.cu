#include <unittest/unittest.h>
#include <thrust/range/algorithm/scatter.h>
#include <thrust/range/algorithm/sequence.h>
#include <thrust/range/algorithm/fill.h>

template <class Vector>
void TestRangeScatterSimple(void)
{
    typedef typename Vector::value_type T;

    Vector map(5);  // scatter indices
    Vector src(5);  // source vector
    Vector dst(8);  // destination vector

    map[0] = 6; map[1] = 3; map[2] = 1; map[3] = 7; map[4] = 2;
    src[0] = 0; src[1] = 1; src[2] = 2; src[3] = 3; src[4] = 4;
    dst[0] = 0; dst[1] = 0; dst[2] = 0; dst[3] = 0; dst[4] = 0; dst[5] = 0; dst[6] = 0; dst[7] = 0;

    using namespace thrust::experimental::range;

    scatter(src, map, dst);

    ASSERT_EQUAL(0, dst[0]);
    ASSERT_EQUAL(2, dst[1]);
    ASSERT_EQUAL(4, dst[2]);
    ASSERT_EQUAL(1, dst[3]);
    ASSERT_EQUAL(0, dst[4]);
    ASSERT_EQUAL(0, dst[5]);
    ASSERT_EQUAL(0, dst[6]);
    ASSERT_EQUAL(3, dst[7]);
}
DECLARE_VECTOR_UNITTEST(TestRangeScatterSimple);


void TestRangeScatterFromHostToDevice(void)
{
    // source vector
    thrust::host_vector<int> src(5);
    src[0] = 0; src[1] = 1; src[2] = 2; src[3] = 3; src[4] = 4;

    // scatter indices
    thrust::host_vector<int> h_map(5);
    h_map[0] = 6; h_map[1] = 3; h_map[2] = 1; h_map[3] = 7; h_map[4] = 2;
    thrust::device_vector<int> d_map = h_map;

    // destination vector
    thrust::device_vector<int> dst(8, (int) 0);

    // expected result
    thrust::device_vector<int> result = dst;
    result[1] = 2;
    result[2] = 4;
    result[3] = 1;
    result[7] = 3;

    using namespace thrust::experimental::range;

    // with map on the device
    scatter(src, d_map, dst);

    ASSERT_EQUAL(result, dst);
}
DECLARE_UNITTEST(TestRangeScatterFromHostToDevice);


void TestRangeScatterFromDeviceToHost(void)
{
    // source vector
    thrust::device_vector<int> src(5);
    src[0] = 0; src[1] = 1; src[2] = 2; src[3] = 3; src[4] = 4;

    // scatter indices
    thrust::host_vector<int> h_map(5);
    h_map[0] = 6; h_map[1] = 3; h_map[2] = 1; h_map[3] = 7; h_map[4] = 2;
    thrust::device_vector<int> d_map = h_map;

    // destination vector
    thrust::host_vector<int> dst(8, (int) 0);

    // expected result
    thrust::host_vector<int> result = dst;
    result[1] = 2;
    result[2] = 4;
    result[3] = 1;
    result[7] = 3;

    using namespace thrust::experimental::range;

    // with map on the host
    scatter(src, h_map, dst);

    ASSERT_EQUAL(result, dst);
}
DECLARE_UNITTEST(TestRangeScatterFromDeviceToHost);


template <typename T>
void TestRangeScatter(const size_t n)
{
    const size_t output_size = std::min((size_t) 10, 2 * n);
    
    thrust::host_vector<T> h_input(n, (T) 1);
    thrust::device_vector<T> d_input(n, (T) 1);
   
    thrust::host_vector<unsigned int> h_map = unittest::random_integers<unsigned int>(n);

    for(size_t i = 0; i < n; i++)
        h_map[i] =  h_map[i] % output_size;
    
    thrust::device_vector<unsigned int> d_map = h_map;

    thrust::host_vector<T>   h_output(output_size, (T) 0);
    thrust::device_vector<T> d_output(output_size, (T) 0);

    using namespace thrust::experimental::range;

    scatter(h_input, h_map, h_output);
    scatter(d_input, d_map, d_output);

    ASSERT_EQUAL(h_output, d_output);
}
DECLARE_VARIABLE_UNITTEST(TestRangeScatter);


template <class Vector>
void TestRangeScatterIfSimple(void)
{
    typedef typename Vector::value_type T;

    Vector flg(5);  // predicate array
    Vector map(5);  // scatter indices
    Vector src(5);  // source vector
    Vector dst(8);  // destination vector

    flg[0] = 0; flg[1] = 1; flg[2] = 0; flg[3] = 1; flg[4] = 0;
    map[0] = 6; map[1] = 3; map[2] = 1; map[3] = 7; map[4] = 2;
    src[0] = 0; src[1] = 1; src[2] = 2; src[3] = 3; src[4] = 4;
    dst[0] = 0; dst[1] = 0; dst[2] = 0; dst[3] = 0; dst[4] = 0; dst[5] = 0; dst[6] = 0; dst[7] = 0;

    using namespace thrust::experimental::range;

    scatter_if(src, map, flg, dst);

    ASSERT_EQUAL(0, dst[0]);
    ASSERT_EQUAL(0, dst[1]);
    ASSERT_EQUAL(0, dst[2]);
    ASSERT_EQUAL(1, dst[3]);
    ASSERT_EQUAL(0, dst[4]);
    ASSERT_EQUAL(0, dst[5]);
    ASSERT_EQUAL(0, dst[6]);
    ASSERT_EQUAL(3, dst[7]);
}
DECLARE_VECTOR_UNITTEST(TestRangeScatterIfSimple);


template <typename T>
class is_even_scatter_if
{
    public:
    __host__ __device__ bool operator()(const T i) const { return (i % 2) == 0; }
};

template <typename T>
void TestRangeScatterIf(const size_t n)
{
    const size_t output_size = std::min((size_t) 10, 2 * n);
    
    thrust::host_vector<T> h_input(n, (T) 1);
    thrust::device_vector<T> d_input(n, (T) 1);
   
    thrust::host_vector<unsigned int> h_map = unittest::random_integers<unsigned int>(n);

    for(size_t i = 0; i < n; i++)
        h_map[i] =  h_map[i] % output_size;
    
    thrust::device_vector<unsigned int> d_map = h_map;

    thrust::host_vector<T>   h_output(output_size, (T) 0);
    thrust::device_vector<T> d_output(output_size, (T) 0);

    thrust::experimental::range::scatter_if(h_input, h_map, h_map, h_output, is_even_scatter_if<unsigned int>());
    thrust::experimental::range::scatter_if(d_input, d_map, d_map, d_output, is_even_scatter_if<unsigned int>());

    ASSERT_EQUAL(h_output, d_output);
}
DECLARE_VARIABLE_UNITTEST(TestRangeScatterIf);


template <typename Vector>
void TestRangeScatterFromSequence(void)
{
    typedef typename Vector::value_type T;

    using namespace thrust::experimental::range;

    Vector source(10);
    sequence(source, 0);

    Vector map(10);
    sequence(map, 0);

    Vector output(10);

    // source has any_space_tag
    fill(output, 0);
    scatter(sequence(0,10),
            map,
            output);

    ASSERT_EQUAL(output, map);
    
    // map has any_space_tag
    fill(output, 0);
    scatter(source,
            sequence(0),
            output);

    ASSERT_EQUAL(output, map);
    
    // source and map have any_space_tag
    fill(output, 0);
    scatter(sequence(0,10),
            sequence(0),
            output);

    ASSERT_EQUAL(output, map);
}
DECLARE_VECTOR_UNITTEST(TestRangeScatterFromSequence);


template <typename Vector>
void TestRangeScatterIfFromSequence(void)
{
    typedef typename Vector::value_type T;

    using namespace thrust::experimental::range;

    Vector source(10);
    sequence(source, 0);

    Vector map(10);
    sequence(map, 0);
    
    Vector stencil(10, 1);

    Vector output(10);

    // source has any_space_tag
    fill(output, 0);
    scatter_if(sequence(0,10),
               map,
               stencil,
               output);

    ASSERT_EQUAL(output, map);
    
    // map has any_space_tag
    fill(output, 0);
    scatter_if(source,
               sequence(0),
               stencil,
               output);

    ASSERT_EQUAL(output, map);
    
    // source and map have any_space_tag
    fill(output, 0);
    scatter_if(sequence(0,10),
               sequence(0),
               stencil,
               output);

    ASSERT_EQUAL(output, map);
}
DECLARE_VECTOR_UNITTEST(TestRangeScatterIfFromSequence);

