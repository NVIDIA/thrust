#include <unittest/unittest.h>
#include <thrust/scan.h>
#include <thrust/functional.h>
#include <thrust/iterator/transform_iterator.h>

struct my_tag : thrust::device_system_tag {};

template <typename Vector>
void TestInclusiveScanByKeySimple(void)
{
    typedef typename Vector::value_type T;
    typedef typename Vector::iterator   Iterator;

    Vector keys(7);
    Vector vals(7);

    Vector output(7, 0);

    keys[0] = 0; vals[0] = 1;
    keys[1] = 1; vals[1] = 2;
    keys[2] = 1; vals[2] = 3;
    keys[3] = 1; vals[3] = 4;
    keys[4] = 2; vals[4] = 5;
    keys[5] = 3; vals[5] = 6;
    keys[6] = 3; vals[6] = 7;

    Iterator iter = thrust::inclusive_scan_by_key(keys.begin(), keys.end(), vals.begin(), output.begin());

    ASSERT_EQUAL_QUIET(iter, output.end());

    ASSERT_EQUAL(output[0],  1);
    ASSERT_EQUAL(output[1],  2);
    ASSERT_EQUAL(output[2],  5);
    ASSERT_EQUAL(output[3],  9);
    ASSERT_EQUAL(output[4],  5);
    ASSERT_EQUAL(output[5],  6);
    ASSERT_EQUAL(output[6], 13);
    
    thrust::inclusive_scan_by_key(keys.begin(), keys.end(), vals.begin(), output.begin(), thrust::equal_to<T>(), thrust::multiplies<T>());

    ASSERT_EQUAL(output[0],  1);
    ASSERT_EQUAL(output[1],  2);
    ASSERT_EQUAL(output[2],  6);
    ASSERT_EQUAL(output[3], 24);
    ASSERT_EQUAL(output[4],  5);
    ASSERT_EQUAL(output[5],  6);
    ASSERT_EQUAL(output[6], 42);
    
    thrust::inclusive_scan_by_key(keys.begin(), keys.end(), vals.begin(), output.begin(), thrust::equal_to<T>());

    ASSERT_EQUAL(output[0],  1);
    ASSERT_EQUAL(output[1],  2);
    ASSERT_EQUAL(output[2],  5);
    ASSERT_EQUAL(output[3],  9);
    ASSERT_EQUAL(output[4],  5);
    ASSERT_EQUAL(output[5],  6);
    ASSERT_EQUAL(output[6], 13);
}
DECLARE_VECTOR_UNITTEST(TestInclusiveScanByKeySimple);


template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator>
OutputIterator inclusive_scan_by_key(my_tag,
                                     InputIterator1,
                                     InputIterator1,
                                     InputIterator2,
                                     OutputIterator result)
{
    *result = 13;
    return result;
}

void TestInclusiveScanByKeyDispatch()
{
    thrust::device_vector<int> vec(1);

    thrust::inclusive_scan_by_key(thrust::retag<my_tag>(vec.begin()),
                                  thrust::retag<my_tag>(vec.begin()),
                                  thrust::retag<my_tag>(vec.begin()),
                                  thrust::retag<my_tag>(vec.begin()));

    ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestInclusiveScanByKeyDispatch);


template <typename Vector>
void TestExclusiveScanByKeySimple(void)
{
    typedef typename Vector::value_type T;
    typedef typename Vector::iterator   Iterator;

    Vector keys(7);
    Vector vals(7);

    Vector output(7, 0);

    keys[0] = 0; vals[0] = 1;
    keys[1] = 1; vals[1] = 2;
    keys[2] = 1; vals[2] = 3;
    keys[3] = 1; vals[3] = 4;
    keys[4] = 2; vals[4] = 5;
    keys[5] = 3; vals[5] = 6;
    keys[6] = 3; vals[6] = 7;
    
    Iterator iter = thrust::exclusive_scan_by_key(keys.begin(), keys.end(), vals.begin(), output.begin());

    ASSERT_EQUAL_QUIET(iter, output.end());

    ASSERT_EQUAL(output[0], 0);
    ASSERT_EQUAL(output[1], 0);
    ASSERT_EQUAL(output[2], 2);
    ASSERT_EQUAL(output[3], 5);
    ASSERT_EQUAL(output[4], 0);
    ASSERT_EQUAL(output[5], 0);
    ASSERT_EQUAL(output[6], 6);

    thrust::exclusive_scan_by_key(keys.begin(), keys.end(), vals.begin(), output.begin(), T(10));

    ASSERT_EQUAL(output[0], 10);
    ASSERT_EQUAL(output[1], 10);
    ASSERT_EQUAL(output[2], 12);
    ASSERT_EQUAL(output[3], 15);
    ASSERT_EQUAL(output[4], 10);
    ASSERT_EQUAL(output[5], 10);
    ASSERT_EQUAL(output[6], 16);
    
    thrust::exclusive_scan_by_key(keys.begin(), keys.end(), vals.begin(), output.begin(), T(10), thrust::equal_to<T>(), thrust::multiplies<T>());

    ASSERT_EQUAL(output[0], 10);
    ASSERT_EQUAL(output[1], 10);
    ASSERT_EQUAL(output[2], 20);
    ASSERT_EQUAL(output[3], 60);
    ASSERT_EQUAL(output[4], 10);
    ASSERT_EQUAL(output[5], 10);
    ASSERT_EQUAL(output[6], 60);
    
    thrust::exclusive_scan_by_key(keys.begin(), keys.end(), vals.begin(), output.begin(), T(10), thrust::equal_to<T>());

    ASSERT_EQUAL(output[0], 10);
    ASSERT_EQUAL(output[1], 10);
    ASSERT_EQUAL(output[2], 12);
    ASSERT_EQUAL(output[3], 15);
    ASSERT_EQUAL(output[4], 10);
    ASSERT_EQUAL(output[5], 10);
    ASSERT_EQUAL(output[6], 16);
}
DECLARE_VECTOR_UNITTEST(TestExclusiveScanByKeySimple);


template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator>
OutputIterator exclusive_scan_by_key(my_tag,
                                     InputIterator1,
                                     InputIterator1,
                                     InputIterator2,
                                     OutputIterator result)
{
    *result = 13;
    return result;
}

void TestExclusiveScanByKeyDispatch()
{
    thrust::device_vector<int> vec(1);

    thrust::exclusive_scan_by_key(thrust::retag<my_tag>(vec.begin()),
                                  thrust::retag<my_tag>(vec.begin()),
                                  thrust::retag<my_tag>(vec.begin()),
                                  thrust::retag<my_tag>(vec.begin()));

    ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestExclusiveScanByKeyDispatch);


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
void TestScanByKeyHeadFlags(void)
{
    typedef typename Vector::value_type T;

    Vector keys(7);
    Vector vals(7);

    Vector output(7, 0);

    keys[0] = 0; vals[0] = 1;
    keys[1] = 1; vals[1] = 2;
    keys[2] = 0; vals[2] = 3;
    keys[3] = 0; vals[3] = 4;
    keys[4] = 1; vals[4] = 5;
    keys[5] = 1; vals[5] = 6;
    keys[6] = 0; vals[6] = 7;
    
    thrust::inclusive_scan_by_key(keys.begin(), keys.end(), vals.begin(), output.begin(), head_flag_predicate(), thrust::plus<T>());

    ASSERT_EQUAL(output[0],  1);
    ASSERT_EQUAL(output[1],  2);
    ASSERT_EQUAL(output[2],  5);
    ASSERT_EQUAL(output[3],  9);
    ASSERT_EQUAL(output[4],  5);
    ASSERT_EQUAL(output[5],  6);
    ASSERT_EQUAL(output[6], 13);

    thrust::exclusive_scan_by_key(keys.begin(), keys.end(), vals.begin(), output.begin(), T(10), head_flag_predicate(), thrust::plus<T>());
    
    ASSERT_EQUAL(output[0], 10);
    ASSERT_EQUAL(output[1], 10);
    ASSERT_EQUAL(output[2], 12);
    ASSERT_EQUAL(output[3], 15);
    ASSERT_EQUAL(output[4], 10);
    ASSERT_EQUAL(output[5], 10);
    ASSERT_EQUAL(output[6], 16);
}
DECLARE_VECTOR_UNITTEST(TestScanByKeyHeadFlags);

template <typename Vector>
void TestInclusiveScanByKeyTransformIterator(void)
{
    typedef typename Vector::value_type T;

    Vector keys(7);
    Vector vals(7);

    Vector output(7, 0);

    keys[0] = 0; vals[0] = 1;
    keys[1] = 1; vals[1] = 2;
    keys[2] = 1; vals[2] = 3;
    keys[3] = 1; vals[3] = 4;
    keys[4] = 2; vals[4] = 5;
    keys[5] = 3; vals[5] = 6;
    keys[6] = 3; vals[6] = 7;

    thrust::inclusive_scan_by_key
        (keys.begin(), keys.end(),
         thrust::make_transform_iterator(vals.begin(), thrust::negate<T>()), 
         output.begin());
    
    ASSERT_EQUAL(output[0],  -1);
    ASSERT_EQUAL(output[1],  -2);
    ASSERT_EQUAL(output[2],  -5);
    ASSERT_EQUAL(output[3],  -9);
    ASSERT_EQUAL(output[4],  -5);
    ASSERT_EQUAL(output[5],  -6);
    ASSERT_EQUAL(output[6], -13);
}
DECLARE_VECTOR_UNITTEST(TestInclusiveScanByKeyTransformIterator);


template <typename Vector>
void TestScanByKeyReusedKeys(void)
{
    typedef typename Vector::value_type T;

    Vector keys(7);
    Vector vals(7);

    Vector output(7, 0);

    keys[0] = 0; vals[0] = 1;
    keys[1] = 1; vals[1] = 2;
    keys[2] = 1; vals[2] = 3;
    keys[3] = 1; vals[3] = 4;
    keys[4] = 0; vals[4] = 5;
    keys[5] = 1; vals[5] = 6;
    keys[6] = 1; vals[6] = 7;
    
    thrust::inclusive_scan_by_key(keys.begin(), keys.end(), vals.begin(), output.begin());

    ASSERT_EQUAL(output[0],  1);
    ASSERT_EQUAL(output[1],  2);
    ASSERT_EQUAL(output[2],  5);
    ASSERT_EQUAL(output[3],  9);
    ASSERT_EQUAL(output[4],  5);
    ASSERT_EQUAL(output[5],  6);
    ASSERT_EQUAL(output[6], 13);

    thrust::exclusive_scan_by_key(keys.begin(), keys.end(), vals.begin(), output.begin(), 10);
    
    ASSERT_EQUAL(output[0], 10);
    ASSERT_EQUAL(output[1], 10);
    ASSERT_EQUAL(output[2], 12);
    ASSERT_EQUAL(output[3], 15);
    ASSERT_EQUAL(output[4], 10);
    ASSERT_EQUAL(output[5], 10);
    ASSERT_EQUAL(output[6], 16);
}
DECLARE_VECTOR_UNITTEST(TestScanByKeyReusedKeys);


template <typename T>
void TestScanByKey(const size_t n)
{
    thrust::host_vector<int> h_keys(n);
    for(size_t i = 0, k = 0; i < n; i++){
        h_keys[i] = k;
        if (rand() % 10 == 0)
            k++;
    }
    thrust::device_vector<int> d_keys = h_keys;

    thrust::host_vector<T>   h_vals = unittest::random_integers<int>(n);
    for(size_t i = 0; i < n; i++)
        h_vals[i] = i % 10;
    thrust::device_vector<T> d_vals = h_vals;

    thrust::host_vector<T>   h_output(n);
    thrust::device_vector<T> d_output(n);
   
    thrust::inclusive_scan_by_key(h_keys.begin(), h_keys.end(), h_vals.begin(), h_output.begin());
    thrust::inclusive_scan_by_key(d_keys.begin(), d_keys.end(), d_vals.begin(), d_output.begin());
    ASSERT_EQUAL(d_output, h_output);
    
    thrust::exclusive_scan_by_key(h_keys.begin(), h_keys.end(), h_vals.begin(), h_output.begin());
    thrust::exclusive_scan_by_key(d_keys.begin(), d_keys.end(), d_vals.begin(), d_output.begin());
    ASSERT_EQUAL(d_output, h_output);
    
    thrust::exclusive_scan_by_key(h_keys.begin(), h_keys.end(), h_vals.begin(), h_output.begin(), (T) 11);
    thrust::exclusive_scan_by_key(d_keys.begin(), d_keys.end(), d_vals.begin(), d_output.begin(), (T) 11);
    ASSERT_EQUAL(d_output, h_output);
    
    // in-place scans
    h_output = h_vals;
    d_output = d_vals;
    thrust::inclusive_scan_by_key(h_keys.begin(), h_keys.end(), h_output.begin(), h_output.begin());
    thrust::inclusive_scan_by_key(d_keys.begin(), d_keys.end(), d_output.begin(), d_output.begin());
    ASSERT_EQUAL(d_output, h_output);
    
    h_output = h_vals;
    d_output = d_vals;
    thrust::exclusive_scan_by_key(h_keys.begin(), h_keys.end(), h_output.begin(), h_output.begin(), (T) 11);
    thrust::exclusive_scan_by_key(d_keys.begin(), d_keys.end(), d_output.begin(), d_output.begin(), (T) 11);
    ASSERT_EQUAL(d_output, h_output);
}
DECLARE_VARIABLE_UNITTEST(TestScanByKey);


void TestScanByKeyMixedTypes(void)
{
    const unsigned int n = 113;
    
    thrust::host_vector<int> h_keys(n);
    for(size_t i = 0, k = 0; i < n; i++){
        h_keys[i] = k;
        if (rand() % 10 == 0)
            k++;
    }
    thrust::device_vector<int> d_keys = h_keys;

    thrust::host_vector<unsigned int> h_vals = unittest::random_integers<unsigned int>(n);
    for(size_t i = 0; i < n; i++)
        h_vals[i] %= 10;
    thrust::device_vector<unsigned int> d_vals = h_vals;

    thrust::host_vector<float>   h_float_output(n);
    thrust::device_vector<float> d_float_output(n);
    thrust::host_vector<int>   h_int_output(n);
    thrust::device_vector<int> d_int_output(n);

    //mixed vals/output types
    thrust::inclusive_scan_by_key(h_keys.begin(), h_keys.end(), h_vals.begin(), h_float_output.begin());
    thrust::inclusive_scan_by_key(d_keys.begin(), d_keys.end(), d_vals.begin(), d_float_output.begin());
    ASSERT_EQUAL(d_float_output, h_float_output);
    
    thrust::exclusive_scan_by_key(h_keys.begin(), h_keys.end(), h_vals.begin(), h_float_output.begin(), (float) 3.5);
    thrust::exclusive_scan_by_key(d_keys.begin(), d_keys.end(), d_vals.begin(), d_float_output.begin(), (float) 3.5);
    ASSERT_EQUAL(d_float_output, h_float_output);
    
    thrust::exclusive_scan_by_key(h_keys.begin(), h_keys.end(), h_vals.begin(), h_float_output.begin(), (int) 3);
    thrust::exclusive_scan_by_key(d_keys.begin(), d_keys.end(), d_vals.begin(), d_float_output.begin(), (int) 3);
    ASSERT_EQUAL(d_float_output, h_float_output);
    
    thrust::exclusive_scan_by_key(h_keys.begin(), h_keys.end(), h_vals.begin(), h_int_output.begin(), (int) 3);
    thrust::exclusive_scan_by_key(d_keys.begin(), d_keys.end(), d_vals.begin(), d_int_output.begin(), (int) 3);
    ASSERT_EQUAL(d_int_output, h_int_output);
    
    thrust::exclusive_scan_by_key(h_keys.begin(), h_keys.end(), h_vals.begin(), h_int_output.begin(), (float) 3.5);
    thrust::exclusive_scan_by_key(d_keys.begin(), d_keys.end(), d_vals.begin(), d_int_output.begin(), (float) 3.5);
    ASSERT_EQUAL(d_int_output, h_int_output);
}
DECLARE_UNITTEST(TestScanByKeyMixedTypes);


void TestScanByKeyLargeInput()
{
    typedef int T;
    const unsigned int N = 1 << 20;

    thrust::host_vector<unsigned int> vals_sizes = unittest::random_integers<unsigned int>(10);
        
    thrust::host_vector<unsigned int>   h_vals = unittest::random_integers<unsigned int>(N);
    thrust::device_vector<unsigned int> d_vals = h_vals;

    thrust::host_vector<unsigned int>   h_output(N, 0);
    thrust::device_vector<unsigned int> d_output(N, 0);

    for (unsigned int i = 0; i < vals_sizes.size(); i++)
    {
        const unsigned int n = vals_sizes[i] % N;

        // define segments
        thrust::host_vector<unsigned int> h_keys(n);
        for(size_t i = 0, k = 0; i < n; i++){
            h_keys[i] = k;
            if (rand() % 100 == 0)
                k++;
        }
        thrust::device_vector<unsigned int> d_keys = h_keys;
    
        thrust::inclusive_scan_by_key(h_keys.begin(), h_keys.begin() + n, h_vals.begin(), h_output.begin());
        thrust::inclusive_scan_by_key(d_keys.begin(), d_keys.begin() + n, d_vals.begin(), d_output.begin());
        ASSERT_EQUAL(d_output, h_output);

        thrust::inclusive_scan_by_key(h_keys.begin(), h_keys.begin() + n, h_vals.begin(), h_output.begin());
        thrust::inclusive_scan_by_key(d_keys.begin(), d_keys.begin() + n, d_vals.begin(), d_output.begin());
        ASSERT_EQUAL(d_output, h_output);
   }
}
DECLARE_UNITTEST(TestScanByKeyLargeInput);


template <typename T, unsigned int N>
void _TestScanByKeyWithLargeTypes(void)
{
    size_t n = (64 * 1024) / sizeof(FixedVector<T,N>);

    thrust::host_vector<   unsigned int   > h_keys(n);
    thrust::host_vector< FixedVector<T,N> > h_vals(n);
    thrust::host_vector< FixedVector<T,N> > h_output(n);

    for(size_t i = 0, k = 0; i < h_vals.size(); i++)
    {
        h_vals[i] = FixedVector<T,N>(i);
        h_keys[i]  = k;
        if (rand() % 5 == 0)
            k++;
    }

    thrust::device_vector<   unsigned int   > d_keys = h_keys;
    thrust::device_vector< FixedVector<T,N> > d_vals = h_vals;
    thrust::device_vector< FixedVector<T,N> > d_output(n);
    
    thrust::inclusive_scan_by_key(h_keys.begin(), h_keys.end(), h_vals.begin(), h_output.begin());
    thrust::inclusive_scan_by_key(d_keys.begin(), d_keys.end(), d_vals.begin(), d_output.begin());

    ASSERT_EQUAL_QUIET(h_output, d_output);
    
    thrust::exclusive_scan_by_key(h_keys.begin(), h_keys.end(), h_vals.begin(), h_output.begin(), FixedVector<T,N>(0));
    thrust::exclusive_scan_by_key(d_keys.begin(), d_keys.end(), d_vals.begin(), d_output.begin(), FixedVector<T,N>(0));
    
    ASSERT_EQUAL_QUIET(h_output, d_output);
}

void TestScanByKeyWithLargeTypes(void)
{
    _TestScanByKeyWithLargeTypes<int,    1>();
    _TestScanByKeyWithLargeTypes<int,    2>();
    _TestScanByKeyWithLargeTypes<int,    4>();
    _TestScanByKeyWithLargeTypes<int,    8>();
    //_TestScanByKeyWithLargeTypes<int,   16>();  // too many resources requested for launch
    //_TestScanByKeyWithLargeTypes<int,   32>();  
    //_TestScanByKeyWithLargeTypes<int,   64>();  // too large to pass as argument
    //_TestScanByKeyWithLargeTypes<int,  128>();
    //_TestScanByKeyWithLargeTypes<int,  256>();
    //_TestScanByKeyWithLargeTypes<int,  512>();
    //_TestScanByKeyWithLargeTypes<int, 1024>();
}
DECLARE_UNITTEST(TestScanByKeyWithLargeTypes);


