#include <unittest/unittest.h>
#include <thrust/range/algorithm/gather.h>
#include <thrust/range/algorithm/sequence.h>
#include <thrust/fill.h>


template <class Vector>
void TestRangeGatherSimple(void)
{
    typedef typename Vector::value_type T;

    Vector map(5);  // gather indices
    Vector src(8);  // source vector
    Vector dst(5);  // destination vector

    map[0] = 6; map[1] = 2; map[2] = 1; map[3] = 7; map[4] = 2;
    src[0] = 0; src[1] = 1; src[2] = 2; src[3] = 3; src[4] = 4; src[5] = 5; src[6] = 6; src[7] = 7;
    dst[0] = 0; dst[1] = 0; dst[2] = 0; dst[3] = 0; dst[4] = 0;

    using namespace thrust::experimental::range;

    gather(map, src, dst);

    ASSERT_EQUAL(6, dst[0]);
    ASSERT_EQUAL(2, dst[1]);
    ASSERT_EQUAL(1, dst[2]);
    ASSERT_EQUAL(7, dst[3]);
    ASSERT_EQUAL(2, dst[4]);
}
DECLARE_VECTOR_UNITTEST(TestRangeGatherSimple);


void TestRangeGatherFromDeviceToHost(void)
{
    // source vector
    thrust::device_vector<int> d_src(8);
    d_src[0] = 0; d_src[1] = 1; d_src[2] = 2; d_src[3] = 3; d_src[4] = 4; d_src[5] = 5; d_src[6] = 6; d_src[7] = 7;

    // gather indices
    thrust::host_vector<int>   h_map(5); 
    h_map[0] = 6; h_map[1] = 2; h_map[2] = 1; h_map[3] = 7; h_map[4] = 2;
    thrust::device_vector<int> d_map = h_map;
   
    // destination vector
    thrust::host_vector<int> h_dst(5, (int) 0);

    using namespace thrust::experimental::range;

    // with map on the device
    gather(d_map, d_src, h_dst);

    ASSERT_EQUAL(6, h_dst[0]);
    ASSERT_EQUAL(2, h_dst[1]);
    ASSERT_EQUAL(1, h_dst[2]);
    ASSERT_EQUAL(7, h_dst[3]);
    ASSERT_EQUAL(2, h_dst[4]);
}
DECLARE_UNITTEST(TestRangeGatherFromDeviceToHost);


void TestRangeGatherFromHostToDevice(void)
{
    // source vector
    thrust::host_vector<int> h_src(8);
    h_src[0] = 0; h_src[1] = 1; h_src[2] = 2; h_src[3] = 3; h_src[4] = 4; h_src[5] = 5; h_src[6] = 6; h_src[7] = 7;

    // gather indices
    thrust::host_vector<int>   h_map(5); 
    h_map[0] = 6; h_map[1] = 2; h_map[2] = 1; h_map[3] = 7; h_map[4] = 2;
    thrust::device_vector<int> d_map = h_map;
   
    // destination vector
    thrust::device_vector<int> d_dst(5, (int) 0);

    using namespace thrust::experimental::range;

    // with map on the host
    gather(h_map, h_src, d_dst);

    ASSERT_EQUAL(6, d_dst[0]);
    ASSERT_EQUAL(2, d_dst[1]);
    ASSERT_EQUAL(1, d_dst[2]);
    ASSERT_EQUAL(7, d_dst[3]);
    ASSERT_EQUAL(2, d_dst[4]);
}
DECLARE_UNITTEST(TestRangeGatherFromHostToDevice);


template <typename T>
void TestRangeGather(const size_t n)
{
    const size_t source_size = std::min((size_t) 10, 2 * n);

    // source vectors to gather from
    thrust::host_vector<T>   h_source = unittest::random_samples<T>(source_size);
    thrust::device_vector<T> d_source = h_source;
  
    // gather indices
    thrust::host_vector<unsigned int> h_map = unittest::random_integers<unsigned int>(n);

    for(size_t i = 0; i < n; i++)
        h_map[i] =  h_map[i] % source_size;
    
    thrust::device_vector<unsigned int> d_map = h_map;

    // gather destination
    thrust::host_vector<T>   h_output(n);
    thrust::device_vector<T> d_output(n);

    using namespace thrust::experimental::range;

    gather(h_map, h_source, h_output);
    gather(d_map, d_source, d_output);

    ASSERT_EQUAL(h_output, d_output);
}
DECLARE_VARIABLE_UNITTEST(TestRangeGather);


template <class Vector>
void TestRangeGatherIfSimple(void)
{
    typedef typename Vector::value_type T;

    Vector flg(5);  // predicate array
    Vector map(5);  // gather indices
    Vector src(8);  // source vector
    Vector dst(5);  // destination vector

    flg[0] = 0; flg[1] = 1; flg[2] = 0; flg[3] = 1; flg[4] = 0;
    map[0] = 6; map[1] = 2; map[2] = 1; map[3] = 7; map[4] = 2;
    src[0] = 0; src[1] = 1; src[2] = 2; src[3] = 3; src[4] = 4; src[5] = 5; src[6] = 6; src[7] = 7;
    dst[0] = 0; dst[1] = 0; dst[2] = 0; dst[3] = 0; dst[4] = 0;

    using namespace thrust::experimental::range;

    gather_if(map, flg, src, dst);

    ASSERT_EQUAL(0, dst[0]);
    ASSERT_EQUAL(2, dst[1]);
    ASSERT_EQUAL(0, dst[2]);
    ASSERT_EQUAL(7, dst[3]);
    ASSERT_EQUAL(0, dst[4]);
}
DECLARE_VECTOR_UNITTEST(TestRangeGatherIfSimple);

template <typename T>
struct is_even_gather_if
{
    __host__ __device__
    bool operator()(const T i) const
    { 
        return (i % 2) == 0;
    }
};

template <typename T>
void TestRangeGatherIf(const size_t n)
{
    const size_t source_size = std::min((size_t) 10, 2 * n);

    // source vectors to gather from
    thrust::host_vector<T>   h_source = unittest::random_samples<T>(source_size);
    thrust::device_vector<T> d_source = h_source;
  
    // gather indices
    thrust::host_vector<unsigned int> h_map = unittest::random_integers<unsigned int>(n);

    for(size_t i = 0; i < n; i++)
        h_map[i] = h_map[i] % source_size;
    
    thrust::device_vector<unsigned int> d_map = h_map;
    
    // gather stencil
    thrust::host_vector<unsigned int> h_stencil = unittest::random_integers<unsigned int>(n);

    for(size_t i = 0; i < n; i++)
        h_stencil[i] = h_stencil[i] % 2;
    
    thrust::device_vector<unsigned int> d_stencil = h_stencil;

    // gather destination
    thrust::host_vector<T>   h_output(n);
    thrust::device_vector<T> d_output(n);

    thrust::experimental::range::gather_if(h_map, h_stencil, h_source, h_output, is_even_gather_if<unsigned int>());
    thrust::experimental::range::gather_if(d_map, d_stencil, d_source, d_output, is_even_gather_if<unsigned int>());

    ASSERT_EQUAL(h_output, d_output);
}
DECLARE_VARIABLE_UNITTEST(TestRangeGatherIf);


template <typename Vector>
void TestRangeGatherFromSequence(void)
{
    typedef typename Vector::value_type T;

    using namespace thrust::experimental::range;

    Vector source(10);
    sequence(source, 0);

    Vector map(10);
    sequence(map, 0);

    Vector output(10);

    // source has any_space_tag
    thrust::fill(output.begin(), output.end(), 0);
    gather(map, sequence(0), output);

    ASSERT_EQUAL(output, map);
    
    // map has any_space_tag
    thrust::fill(output.begin(), output.end(), 0);
    gather(sequence(0,(int)source.size()), source, output);

    ASSERT_EQUAL(output, map);
    
    // source and map have any_space_tag
    thrust::fill(output.begin(), output.end(), 0);
    gather(sequence(0,(int)output.size()),
           sequence(0),
           output);

    ASSERT_EQUAL(output, map);
}
DECLARE_VECTOR_UNITTEST(TestRangeGatherFromSequence);

