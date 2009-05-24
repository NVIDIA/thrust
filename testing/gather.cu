#include <thrusttest/unittest.h>
#include <thrust/gather.h>

#include <thrust/fill.h>

template <class Vector>
void TestGatherSimple(void)
{
    typedef typename Vector::value_type T;

    Vector map(5);  // gather indices
    Vector src(8);  // source vector
    Vector dst(5);  // destination vector

    map[0] = 6; map[1] = 2; map[2] = 1; map[3] = 7; map[4] = 2;
    src[0] = 0; src[1] = 1; src[2] = 2; src[3] = 3; src[4] = 4; src[5] = 5; src[6] = 6; src[7] = 7;
    dst[0] = 0; dst[1] = 0; dst[2] = 0; dst[3] = 0; dst[4] = 0;

    thrust::gather(dst.begin(), dst.end(), map.begin(), src.begin());

    ASSERT_EQUAL(dst[0], 6);
    ASSERT_EQUAL(dst[1], 2);
    ASSERT_EQUAL(dst[2], 1);
    ASSERT_EQUAL(dst[3], 7);
    ASSERT_EQUAL(dst[4], 2);
}
DECLARE_VECTOR_UNITTEST(TestGatherSimple);


void TestGatherFromDeviceToHost(void)
{
    // source vector
    thrust::device_vector<int> src(8);
    src[0] = 0; src[1] = 1; src[2] = 2; src[3] = 3; src[4] = 4; src[5] = 5; src[6] = 6; src[7] = 7;

    // gather indices
    thrust::host_vector<int>   h_map(5); 
    h_map[0] = 6; h_map[1] = 2; h_map[2] = 1; h_map[3] = 7; h_map[4] = 2;
    thrust::device_vector<int> d_map = h_map;
   
    // destination vector
    thrust::host_vector<int> dst(5, (int) 0);

    //// with map on the device
    thrust::gather(dst.begin(), dst.end(), d_map.begin(), src.begin());

    ASSERT_EQUAL(dst[0], 6);
    ASSERT_EQUAL(dst[1], 2);
    ASSERT_EQUAL(dst[2], 1);
    ASSERT_EQUAL(dst[3], 7);
    ASSERT_EQUAL(dst[4], 2);

    // clear the destination vector
    thrust::fill(dst.begin(), dst.end(), (int) 0);
    
    // with map on the host
    thrust::gather(dst.begin(), dst.end(), h_map.begin(), src.begin());

    ASSERT_EQUAL(dst[0], 6);
    ASSERT_EQUAL(dst[1], 2);
    ASSERT_EQUAL(dst[2], 1);
    ASSERT_EQUAL(dst[3], 7);
    ASSERT_EQUAL(dst[4], 2);
}
DECLARE_UNITTEST(TestGatherFromDeviceToHost);


void TestGatherFromHostToDevice(void)
{
    // source vector
    thrust::host_vector<int> src(8);
    src[0] = 0; src[1] = 1; src[2] = 2; src[3] = 3; src[4] = 4; src[5] = 5; src[6] = 6; src[7] = 7;

    // gather indices
    thrust::host_vector<int>   h_map(5); 
    h_map[0] = 6; h_map[1] = 2; h_map[2] = 1; h_map[3] = 7; h_map[4] = 2;
    thrust::device_vector<int> d_map = h_map;
   
    // destination vector
    thrust::device_vector<int> dst(5, (int) 0);

    //// with map on the device
    thrust::gather(dst.begin(), dst.end(), d_map.begin(), src.begin());

    ASSERT_EQUAL(dst[0], 6);
    ASSERT_EQUAL(dst[1], 2);
    ASSERT_EQUAL(dst[2], 1);
    ASSERT_EQUAL(dst[3], 7);
    ASSERT_EQUAL(dst[4], 2);

    // clear the destination vector
    thrust::fill(dst.begin(), dst.end(), (int) 0);
    
    // with map on the host
    thrust::gather(dst.begin(), dst.end(), h_map.begin(), src.begin());

    ASSERT_EQUAL(dst[0], 6);
    ASSERT_EQUAL(dst[1], 2);
    ASSERT_EQUAL(dst[2], 1);
    ASSERT_EQUAL(dst[3], 7);
    ASSERT_EQUAL(dst[4], 2);
}
DECLARE_UNITTEST(TestGatherFromHostToDevice);


template <typename T>
void TestGather(const size_t n)
{
    const size_t source_size = std::min((size_t) 10, 2 * n);

    // source vectors to gather from
    thrust::host_vector<T>   h_source = thrusttest::random_samples<T>(source_size);
    thrust::device_vector<T> d_source = h_source;
  
    // gather indices
    thrust::host_vector<unsigned int> h_map = thrusttest::random_integers<unsigned int>(n);

    for(size_t i = 0; i < n; i++)
        h_map[i] =  h_map[i] % source_size;
    
    thrust::device_vector<unsigned int> d_map = h_map;

    // gather destination
    thrust::host_vector<T>   h_output(n);
    thrust::device_vector<T> d_output(n);

    thrust::gather(h_output.begin(), h_output.end(), h_map.begin(), h_source.begin());
    thrust::gather(d_output.begin(), d_output.end(), d_map.begin(), d_source.begin());

    ASSERT_EQUAL(h_output, d_output);
}
DECLARE_VARIABLE_UNITTEST(TestGather);


template <class Vector>
void TestGatherIfSimple(void)
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

    thrust::gather_if(dst.begin(), dst.end(), map.begin(), flg.begin(), src.begin());

    ASSERT_EQUAL(dst[0], 0);
    ASSERT_EQUAL(dst[1], 2);
    ASSERT_EQUAL(dst[2], 0);
    ASSERT_EQUAL(dst[3], 7);
    ASSERT_EQUAL(dst[4], 0);
}
DECLARE_VECTOR_UNITTEST(TestGatherIfSimple);



template <typename T>
class is_even_gather_if
{
    public:
    __host__ __device__ bool operator()(const T i) const { return (i % 2) == 1; }
};

template <typename T>
void TestGatherIf(const size_t n)
{
    const size_t source_size = std::min((size_t) 10, 2 * n);

    // source vectors to gather from
    thrust::host_vector<T>   h_source = thrusttest::random_samples<T>(source_size);
    thrust::device_vector<T> d_source = h_source;
  
    // gather indices
    thrust::host_vector<unsigned int> h_map = thrusttest::random_integers<unsigned int>(n);

    for(size_t i = 0; i < n; i++)
        h_map[i] = h_map[i] % source_size;
    
    thrust::device_vector<unsigned int> d_map = h_map;
    
    // gather stencil
    thrust::host_vector<unsigned int> h_stencil = thrusttest::random_integers<unsigned int>(n);

    for(size_t i = 0; i < n; i++)
        h_stencil[i] = h_stencil[i] % 2;
    
    thrust::device_vector<unsigned int> d_stencil = h_stencil;

    // gather destination
    thrust::host_vector<T>   h_output(n);
    thrust::device_vector<T> d_output(n);

    thrust::gather_if(h_output.begin(), h_output.end(), h_map.begin(), h_stencil.begin(), h_source.begin(), is_even_gather_if<unsigned int>());
    thrust::gather_if(d_output.begin(), d_output.end(), d_map.begin(), d_stencil.begin(), d_source.begin(), is_even_gather_if<unsigned int>());

    ASSERT_EQUAL(h_output, d_output);
}
DECLARE_VARIABLE_UNITTEST(TestGatherIf);

