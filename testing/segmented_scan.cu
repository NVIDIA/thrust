#include <thrusttest/unittest.h>
#include <thrust/segmented_scan.h>
#include <thrust/functional.h>
#include <thrust/iterator/transform_iterator.h>

template <typename Vector>
void TestInclusiveSegmentedScanSimple(void)
{
    typedef typename Vector::value_type T;

    Vector input(7);
    Vector key(7);

    Vector output(7, 0);

    input[0] = 1;  key[0] = 0;
    input[1] = 2;  key[1] = 1;
    input[2] = 3;  key[2] = 1;
    input[3] = 4;  key[3] = 1;
    input[4] = 5;  key[4] = 2;
    input[5] = 6;  key[5] = 3;
    input[6] = 7;  key[6] = 3;

    thrust::experimental::inclusive_segmented_scan(input.begin(), input.end(), key.begin(), output.begin());

    ASSERT_EQUAL(output[0],  1);
    ASSERT_EQUAL(output[1],  2);
    ASSERT_EQUAL(output[2],  5);
    ASSERT_EQUAL(output[3],  9);
    ASSERT_EQUAL(output[4],  5);
    ASSERT_EQUAL(output[5],  6);
    ASSERT_EQUAL(output[6], 13);
    
    thrust::experimental::inclusive_segmented_scan(input.begin(), input.end(), key.begin(), output.begin(), thrust::multiplies<T>());

    ASSERT_EQUAL(output[0],  1);
    ASSERT_EQUAL(output[1],  2);
    ASSERT_EQUAL(output[2],  6);
    ASSERT_EQUAL(output[3], 24);
    ASSERT_EQUAL(output[4],  5);
    ASSERT_EQUAL(output[5],  6);
    ASSERT_EQUAL(output[6], 42);
    
    thrust::experimental::inclusive_segmented_scan(input.begin(), input.end(), key.begin(), output.begin(), thrust::plus<T>(), thrust::equal_to<T>());

    ASSERT_EQUAL(output[0],  1);
    ASSERT_EQUAL(output[1],  2);
    ASSERT_EQUAL(output[2],  5);
    ASSERT_EQUAL(output[3],  9);
    ASSERT_EQUAL(output[4],  5);
    ASSERT_EQUAL(output[5],  6);
    ASSERT_EQUAL(output[6], 13);
    
}
DECLARE_VECTOR_UNITTEST(TestInclusiveSegmentedScanSimple);


template <typename Vector>
void TestExclusiveSegmentedScanSimple(void)
{
    typedef typename Vector::value_type T;

    Vector input(7);
    Vector key(7);

    Vector output(7, 0);

    input[0] = 1;  key[0] = 0;
    input[1] = 2;  key[1] = 1;
    input[2] = 3;  key[2] = 1;
    input[3] = 4;  key[3] = 1;
    input[4] = 5;  key[4] = 2;
    input[5] = 6;  key[5] = 3;
    input[6] = 7;  key[6] = 3;

    thrust::experimental::exclusive_segmented_scan(input.begin(), input.end(), key.begin(), output.begin(), 10);

    ASSERT_EQUAL(output[0], 10);
    ASSERT_EQUAL(output[1], 10);
    ASSERT_EQUAL(output[2], 12);
    ASSERT_EQUAL(output[3], 15);
    ASSERT_EQUAL(output[4], 10);
    ASSERT_EQUAL(output[5], 10);
    ASSERT_EQUAL(output[6], 16);
    
    thrust::experimental::exclusive_segmented_scan(input.begin(), input.end(), key.begin(), output.begin(), 10, thrust::multiplies<T>());

    ASSERT_EQUAL(output[0], 10);
    ASSERT_EQUAL(output[1], 10);
    ASSERT_EQUAL(output[2], 20);
    ASSERT_EQUAL(output[3], 60);
    ASSERT_EQUAL(output[4], 10);
    ASSERT_EQUAL(output[5], 10);
    ASSERT_EQUAL(output[6], 60);
    
    thrust::experimental::exclusive_segmented_scan(input.begin(), input.end(), key.begin(), output.begin(), 10, thrust::plus<T>(), thrust::equal_to<T>());

    ASSERT_EQUAL(output[0], 10);
    ASSERT_EQUAL(output[1], 10);
    ASSERT_EQUAL(output[2], 12);
    ASSERT_EQUAL(output[3], 15);
    ASSERT_EQUAL(output[4], 10);
    ASSERT_EQUAL(output[5], 10);
    ASSERT_EQUAL(output[6], 16);
}
DECLARE_VECTOR_UNITTEST(TestExclusiveSegmentedScanSimple);

struct head_flag_predicate
{
    template <typename T>
    __host__ __device__
    bool operator()(const T& a, const T& b)
    {
        return b ? false : true;
    }
};


template <typename Vector>
void TestSegmentedScanHeadFlags(void)
{
#ifdef __APPLE__
    // nvcc has trouble with the head_flag_predicate struct it seems
    KNOWN_FAILURE
#else
    typedef typename Vector::value_type T;

    Vector input(7);
    Vector key(7);

    Vector output(7, 0);

    input[0] = 1;  key[0] = 0;
    input[1] = 2;  key[1] = 1;
    input[2] = 3;  key[2] = 0;
    input[3] = 4;  key[3] = 0;
    input[4] = 5;  key[4] = 1;
    input[5] = 6;  key[5] = 1;
    input[6] = 7;  key[6] = 0;
    
    thrust::experimental::inclusive_segmented_scan(input.begin(), input.end(), key.begin(), output.begin(), thrust::plus<T>(), head_flag_predicate());

    ASSERT_EQUAL(output[0],  1);
    ASSERT_EQUAL(output[1],  2);
    ASSERT_EQUAL(output[2],  5);
    ASSERT_EQUAL(output[3],  9);
    ASSERT_EQUAL(output[4],  5);
    ASSERT_EQUAL(output[5],  6);
    ASSERT_EQUAL(output[6], 13);

    thrust::experimental::exclusive_segmented_scan(input.begin(), input.end(), key.begin(), output.begin(), 10, thrust::plus<T>(), head_flag_predicate());
    
    ASSERT_EQUAL(output[0], 10);
    ASSERT_EQUAL(output[1], 10);
    ASSERT_EQUAL(output[2], 12);
    ASSERT_EQUAL(output[3], 15);
    ASSERT_EQUAL(output[4], 10);
    ASSERT_EQUAL(output[5], 10);
    ASSERT_EQUAL(output[6], 16);
#endif
}
DECLARE_VECTOR_UNITTEST(TestSegmentedScanHeadFlags);


template <typename Vector>
void TestInclusiveSegmentedScanTransformIterator(void)
{
    typedef typename Vector::value_type T;

    Vector input(7);
    Vector key(7);

    Vector output(7, 0);

    input[0] = 1;  key[0] = 0;
    input[1] = 2;  key[1] = 1;
    input[2] = 3;  key[2] = 1;
    input[3] = 4;  key[3] = 1;
    input[4] = 5;  key[4] = 2;
    input[5] = 6;  key[5] = 3;
    input[6] = 7;  key[6] = 3;

    thrust::experimental::inclusive_segmented_scan(
        thrust::experimental::make_transform_iterator(input.begin(), thrust::negate<T>()), 
        thrust::experimental::make_transform_iterator(input.end(),   thrust::negate<T>()), 
        key.begin(), output.begin());
    
    ASSERT_EQUAL(output[0],  -1);
    ASSERT_EQUAL(output[1],  -2);
    ASSERT_EQUAL(output[2],  -5);
    ASSERT_EQUAL(output[3],  -9);
    ASSERT_EQUAL(output[4],  -5);
    ASSERT_EQUAL(output[5],  -6);
    ASSERT_EQUAL(output[6], -13);
}
DECLARE_VECTOR_UNITTEST(TestInclusiveSegmentedScanTransformIterator);


template <typename Vector>
void TestSegmentedScanReusedKeys(void)
{
    typedef typename Vector::value_type T;

    Vector input(7);
    Vector key(7);

    Vector output(7, 0);

    input[0] = 1;  key[0] = 0;
    input[1] = 2;  key[1] = 1;
    input[2] = 3;  key[2] = 1;
    input[3] = 4;  key[3] = 1;
    input[4] = 5;  key[4] = 0;
    input[5] = 6;  key[5] = 1;
    input[6] = 7;  key[6] = 1;
    
    thrust::experimental::inclusive_segmented_scan(input.begin(), input.end(), key.begin(), output.begin());

    ASSERT_EQUAL(output[0],  1);
    ASSERT_EQUAL(output[1],  2);
    ASSERT_EQUAL(output[2],  5);
    ASSERT_EQUAL(output[3],  9);
    ASSERT_EQUAL(output[4],  5);
    ASSERT_EQUAL(output[5],  6);
    ASSERT_EQUAL(output[6], 13);

    thrust::experimental::exclusive_segmented_scan(input.begin(), input.end(), key.begin(), output.begin(), 10);
    
    ASSERT_EQUAL(output[0], 10);
    ASSERT_EQUAL(output[1], 10);
    ASSERT_EQUAL(output[2], 12);
    ASSERT_EQUAL(output[3], 15);
    ASSERT_EQUAL(output[4], 10);
    ASSERT_EQUAL(output[5], 10);
    ASSERT_EQUAL(output[6], 16);
}
DECLARE_VECTOR_UNITTEST(TestSegmentedScanReusedKeys);


template <typename T>
void TestSegmentedScan(const size_t n)
{
    thrust::host_vector<T>   h_input = thrusttest::random_integers<int>(n);
    for(size_t i = 0; i < n; i++)
        h_input[i] = i % 10;
    thrust::device_vector<T> d_input = h_input;

    thrust::host_vector<T>   h_output(n);
    thrust::device_vector<T> d_output(n);
    
    thrust::host_vector<int> h_keys(n);
    for(size_t i = 0, k = 0; i < n; i++){
        h_keys[i] = k;
        if (rand() % 10 == 0)
            k++;
    }
    thrust::device_vector<int> d_keys = h_keys;
   
//    for(size_t i = 0; i < min(80, int(n)); i++)
//        std::cout << i << "  " << h_keys[i] << " " << h_input[i] << std::endl;

    thrust::experimental::inclusive_segmented_scan(h_input.begin(), h_input.end(), h_keys.begin(), h_output.begin());
    thrust::experimental::inclusive_segmented_scan(d_input.begin(), d_input.end(), d_keys.begin(), d_output.begin());
    ASSERT_EQUAL(d_output, h_output);
    
    thrust::experimental::exclusive_segmented_scan(h_input.begin(), h_input.end(), h_keys.begin(), h_output.begin());
    thrust::experimental::exclusive_segmented_scan(d_input.begin(), d_input.end(), d_keys.begin(), d_output.begin());
    ASSERT_EQUAL(d_output, h_output);
    
    thrust::experimental::exclusive_segmented_scan(h_input.begin(), h_input.end(), h_keys.begin(), h_output.begin(), (T) 11);
    thrust::experimental::exclusive_segmented_scan(d_input.begin(), d_input.end(), d_keys.begin(), d_output.begin(), (T) 11);
    ASSERT_EQUAL(d_output, h_output);
    
    // in-place scans
    h_output = h_input;
    d_output = d_input;
    thrust::experimental::inclusive_segmented_scan(h_output.begin(), h_output.end(), h_keys.begin(), h_output.begin());
    thrust::experimental::inclusive_segmented_scan(d_output.begin(), d_output.end(), d_keys.begin(), d_output.begin());
    ASSERT_EQUAL(d_output, h_output);
    
    h_output = h_input;
    d_output = d_input;
    thrust::experimental::exclusive_segmented_scan(h_output.begin(), h_output.end(), h_keys.begin(), h_output.begin(), (T) 11);
    thrust::experimental::exclusive_segmented_scan(d_output.begin(), d_output.end(), d_keys.begin(), d_output.begin(), (T) 11);
    ASSERT_EQUAL(d_output, h_output);
}
DECLARE_VARIABLE_UNITTEST(TestSegmentedScan);


void TestSegmentedScanMixedTypes(void)
{
    const unsigned int n = 113;

    thrust::host_vector<unsigned int> h_input = thrusttest::random_integers<unsigned int>(n);
    for(size_t i = 0; i < n; i++)
        h_input[i] %= 10;
    thrust::device_vector<unsigned int> d_input = h_input;
    
    thrust::host_vector<int> h_keys(n);
    for(size_t i = 0, k = 0; i < n; i++){
        h_keys[i] = k;
        if (rand() % 10 == 0)
            k++;
    }
    thrust::device_vector<int> d_keys = h_keys;

    thrust::host_vector<float>   h_float_output(n);
    thrust::device_vector<float> d_float_output(n);
    thrust::host_vector<int>   h_int_output(n);
    thrust::device_vector<int> d_int_output(n);

    //mixed input/output types
    thrust::experimental::inclusive_segmented_scan(h_input.begin(), h_input.end(), h_keys.begin(), h_float_output.begin());
    thrust::experimental::inclusive_segmented_scan(d_input.begin(), d_input.end(), d_keys.begin(), d_float_output.begin());
    ASSERT_EQUAL(d_float_output, h_float_output);
    
    thrust::experimental::exclusive_segmented_scan(h_input.begin(), h_input.end(), h_keys.begin(), h_float_output.begin(), (float) 3.5);
    thrust::experimental::exclusive_segmented_scan(d_input.begin(), d_input.end(), d_keys.begin(), d_float_output.begin(), (float) 3.5);
    ASSERT_EQUAL(d_float_output, h_float_output);
    
    thrust::experimental::exclusive_segmented_scan(h_input.begin(), h_input.end(), h_keys.begin(), h_float_output.begin(), (int) 3);
    thrust::experimental::exclusive_segmented_scan(d_input.begin(), d_input.end(), d_keys.begin(), d_float_output.begin(), (int) 3);
    ASSERT_EQUAL(d_float_output, h_float_output);
    
    thrust::experimental::exclusive_segmented_scan(h_input.begin(), h_input.end(), h_keys.begin(), h_int_output.begin(), (int) 3);
    thrust::experimental::exclusive_segmented_scan(d_input.begin(), d_input.end(), d_keys.begin(), d_int_output.begin(), (int) 3);
    ASSERT_EQUAL(d_int_output, h_int_output);
    
    thrust::experimental::exclusive_segmented_scan(h_input.begin(), h_input.end(), h_keys.begin(), h_int_output.begin(), (float) 3.5);
    thrust::experimental::exclusive_segmented_scan(d_input.begin(), d_input.end(), d_keys.begin(), d_int_output.begin(), (float) 3.5);
    ASSERT_EQUAL(d_int_output, h_int_output);
}
DECLARE_UNITTEST(TestSegmentedScanMixedTypes);


void TestSegmentedScanLargeInput()
{
    typedef int T;
    const unsigned int N = 1 << 20;
    const unsigned int K = 100;

    thrust::host_vector<unsigned int> input_sizes = thrusttest::random_integers<unsigned int>(10);
        
    thrust::host_vector<unsigned int>   h_input = thrusttest::random_integers<unsigned int>(N);
    thrust::device_vector<unsigned int> d_input = h_input;

    thrust::host_vector<unsigned int>   h_output(N, 0);
    thrust::device_vector<unsigned int> d_output(N, 0);

    for (unsigned int i = 0; i < input_sizes.size(); i++)
    {
        const unsigned int n = input_sizes[i] % N;
        const unsigned int k = input_sizes[i] % K;

        // define segments
        thrust::host_vector<unsigned int> h_keys(n);
        for(size_t i = 0, k = 0; i < n; i++){
            h_keys[i] = k;
            if (rand() % 100 == 0)
                k++;
        }
        thrust::device_vector<unsigned int> d_keys = h_keys;
    
        thrust::experimental::inclusive_segmented_scan(h_input.begin(), h_input.begin() + n, h_keys.begin(), h_output.begin());
        thrust::experimental::inclusive_segmented_scan(d_input.begin(), d_input.begin() + n, d_keys.begin(), d_output.begin());
        ASSERT_EQUAL(d_output, h_output);

        thrust::experimental::inclusive_segmented_scan(h_input.begin(), h_input.begin() + n, h_keys.begin(), h_output.begin());
        thrust::experimental::inclusive_segmented_scan(d_input.begin(), d_input.begin() + n, d_keys.begin(), d_output.begin());
        ASSERT_EQUAL(d_output, h_output);
   }
}
DECLARE_UNITTEST(TestSegmentedScanLargeInput);


template <typename T, unsigned int N>
void _TestSegmentedScanWithLargeTypes(void)
{
    size_t n = (64 * 1024) / sizeof(FixedVector<T,N>);

    thrust::host_vector< FixedVector<T,N> > h_input(n);
    thrust::host_vector< FixedVector<T,N> > h_output(n);
    thrust::host_vector<   unsigned int   > h_keys(n);

    for(size_t i = 0, k = 0; i < h_input.size(); i++)
    {
        h_input[i] = FixedVector<T,N>(i);
        h_keys[i]  = k;
        if (rand() % 5 == 0)
            k++;
    }

    thrust::device_vector< FixedVector<T,N> > d_input = h_input;
    thrust::device_vector< FixedVector<T,N> > d_output(n);
    thrust::device_vector<   unsigned int   > d_keys = h_keys;
    
    thrust::experimental::inclusive_segmented_scan(h_input.begin(), h_input.end(), h_keys.begin(), h_output.begin());
    thrust::experimental::inclusive_segmented_scan(d_input.begin(), d_input.end(), d_keys.begin(), d_output.begin());

    ASSERT_EQUAL_QUIET(h_output, d_output);
    
    thrust::experimental::exclusive_segmented_scan(h_input.begin(), h_input.end(), h_keys.begin(), h_output.begin(), FixedVector<T,N>(0));
    thrust::experimental::exclusive_segmented_scan(d_input.begin(), d_input.end(), d_keys.begin(), d_output.begin(), FixedVector<T,N>(0));
    
    ASSERT_EQUAL_QUIET(h_output, d_output);
}

void TestSegmentedScanWithLargeTypes(void)
{
    _TestSegmentedScanWithLargeTypes<int,    1>();
    _TestSegmentedScanWithLargeTypes<int,    2>();
    _TestSegmentedScanWithLargeTypes<int,    4>();
    _TestSegmentedScanWithLargeTypes<int,    8>();
    //_TestSegmentedScanWithLargeTypes<int,   16>();  // too many resources requested for launch
    //_TestSegmentedScanWithLargeTypes<int,   32>();  
    //_TestSegmentedScanWithLargeTypes<int,   64>();  // too large to pass as argument
    //_TestSegmentedScanWithLargeTypes<int,  128>();
    //_TestSegmentedScanWithLargeTypes<int,  256>();
    //_TestSegmentedScanWithLargeTypes<int,  512>();
    //_TestSegmentedScanWithLargeTypes<int, 1024>();
}
DECLARE_UNITTEST(TestSegmentedScanWithLargeTypes);



