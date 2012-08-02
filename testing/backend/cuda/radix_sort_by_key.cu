#include <unittest/unittest.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>
#include <thrust/device_malloc_allocator.h>

#include <thrust/sort.h>
#include <thrust/system/cuda/detail/detail/stable_radix_sort.h>

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA

using namespace unittest;

template <class Vector>
void InitializeSimpleKeyRadixSortTest(Vector& unsorted_keys, Vector& sorted_keys)
{
    unsorted_keys.resize(7);
    unsorted_keys[0] = 1; 
    unsorted_keys[1] = 3; 
    unsorted_keys[2] = 6;
    unsorted_keys[3] = 5;
    unsorted_keys[4] = 2;
    unsorted_keys[5] = 0;
    unsorted_keys[6] = 4;

    sorted_keys.resize(7); 
    sorted_keys[0] = 0; 
    sorted_keys[1] = 1; 
    sorted_keys[2] = 2;
    sorted_keys[3] = 3;
    sorted_keys[4] = 4;
    sorted_keys[5] = 5;
    sorted_keys[6] = 6;
}

template <class Vector>
void InitializeSimpleKeyValueRadixSortTest(Vector& unsorted_keys, Vector& unsorted_values,
                                           Vector& sorted_keys,   Vector& sorted_values)
{
    unsorted_keys.resize(7);   
    unsorted_values.resize(7);   
    unsorted_keys[0] = 1;  unsorted_values[0] = 0;
    unsorted_keys[1] = 3;  unsorted_values[1] = 1;
    unsorted_keys[2] = 6;  unsorted_values[2] = 2;
    unsorted_keys[3] = 5;  unsorted_values[3] = 3;
    unsorted_keys[4] = 2;  unsorted_values[4] = 4;
    unsorted_keys[5] = 0;  unsorted_values[5] = 5;
    unsorted_keys[6] = 4;  unsorted_values[6] = 6;
    
    sorted_keys.resize(7);
    sorted_values.resize(7);
    sorted_keys[0] = 0;  sorted_values[1] = 0;  
    sorted_keys[1] = 1;  sorted_values[3] = 1;  
    sorted_keys[2] = 2;  sorted_values[6] = 2;
    sorted_keys[3] = 3;  sorted_values[5] = 3;
    sorted_keys[4] = 4;  sorted_values[2] = 4;
    sorted_keys[5] = 5;  sorted_values[0] = 5;
    sorted_keys[6] = 6;  sorted_values[4] = 6;
}

template <class Vector>
struct TestRadixSortKeyValueSimple
{
  void operator()(const size_t dummy)
  {
    Vector unsorted_keys, unsorted_values;
    Vector   sorted_keys,   sorted_values;

    InitializeSimpleKeyValueRadixSortTest(unsorted_keys, unsorted_values, sorted_keys, sorted_values);

    thrust::cuda::tag cuda_tag;
    thrust::system::cuda::detail::detail::stable_radix_sort_by_key(cuda_tag, unsorted_keys.begin(), unsorted_keys.end(), unsorted_values.begin());

    ASSERT_EQUAL(unsorted_keys,   sorted_keys);
    ASSERT_EQUAL(unsorted_values, sorted_values);
  }
};
VectorUnitTest<TestRadixSortKeyValueSimple, ThirtyTwoBitTypes, thrust::device_vector, thrust::device_malloc_allocator> TestRadixSortKeyValueSimpleDeviceInstance;


typedef unittest::type_list<
#if !(defined(__GNUC__) && (__GNUC__ <= 4) && (__GNUC_MINOR__ <= 1))
// XXX GCC 4.1 miscompiles the char sorts with -O2 for some reason
                            char,
                            signed char,
                            unsigned char,
#endif
                            short,
                            unsigned short,
                            int,
                            unsigned int,
                            long,
                            unsigned long,
                            long long,
                            unsigned long long,
                            float,
                            double> RadixSortKeyTypes;

template <typename T>
struct TestRadixSortByKey
{
  void operator()(const size_t n)
  {
    thrust::host_vector<T>   h_keys = unittest::random_integers<T>(n);
    thrust::device_vector<T> d_keys = h_keys;

    thrust::host_vector<unsigned int>   h_values(n);
    thrust::device_vector<unsigned int> d_values(n);
    thrust::sequence(h_values.begin(), h_values.end());
    thrust::sequence(d_values.begin(), d_values.end());

    thrust::stable_sort_by_key(h_keys.begin(), h_keys.end(), h_values.begin());

    thrust::cuda::tag cuda_tag;
    thrust::system::cuda::detail::detail::stable_radix_sort_by_key(cuda_tag, d_keys.begin(), d_keys.end(), d_values.begin());

    ASSERT_ALMOST_EQUAL(h_keys, d_keys);
    ASSERT_ALMOST_EQUAL(h_values, d_values);
  }
};
VariableUnitTest<TestRadixSortByKey, RadixSortKeyTypes> TestRadixSortByKeyInstance;

#endif // THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA

