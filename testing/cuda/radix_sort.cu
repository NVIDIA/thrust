#include <unittest/unittest.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>
#include <thrust/device_malloc_allocator.h>

#include <thrust/sort.h>
#include <thrust/detail/device/cuda/detail/stable_radix_sort.h>

#if THRUST_DEVICE_BACKEND == THRUST_DEVICE_BACKEND_CUDA

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
void InitializeSimpleStableKeyRadixSortTest(Vector& unsorted_keys, Vector& sorted_keys)
{
    unsorted_keys.resize(9);   
    unsorted_keys[0] = 25; 
    unsorted_keys[1] = 14; 
    unsorted_keys[2] = 35; 
    unsorted_keys[3] = 16; 
    unsorted_keys[4] = 26; 
    unsorted_keys[5] = 34; 
    unsorted_keys[6] = 36; 
    unsorted_keys[7] = 24; 
    unsorted_keys[8] = 15; 
    
    sorted_keys.resize(9);
    sorted_keys[0] = 14; 
    sorted_keys[1] = 16; 
    sorted_keys[2] = 15; 
    sorted_keys[3] = 25; 
    sorted_keys[4] = 26; 
    sorted_keys[5] = 24; 
    sorted_keys[6] = 35; 
    sorted_keys[7] = 34; 
    sorted_keys[8] = 36; 
}


template <class Vector>
struct TestRadixSortKeySimple
{
  void operator()(const size_t dummy)
  {
    typedef typename Vector::value_type T;

    Vector unsorted_keys;
    Vector   sorted_keys;

    InitializeSimpleKeyRadixSortTest(unsorted_keys, sorted_keys);

    thrust::detail::device::cuda::detail::stable_radix_sort(unsorted_keys.begin(), unsorted_keys.end());

    ASSERT_EQUAL(unsorted_keys, sorted_keys);
  }
};
VectorUnitTest<TestRadixSortKeySimple, ThirtyTwoBitTypes, thrust::device_vector, thrust::device_malloc_allocator> TestRadixSortKeySimpleDeviceInstance;


//still need to do long/ulong and maybe double

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
                            double> RadixSortKeyTypes;

template <typename T>
struct TestRadixSort
{
  void operator()(const size_t n)
  {
    thrust::host_vector<T>   h_keys = unittest::random_integers<T>(n);
    thrust::device_vector<T> d_keys = h_keys;

    thrust::stable_sort(h_keys.begin(), h_keys.end());
    thrust::detail::device::cuda::detail::stable_radix_sort(d_keys.begin(), d_keys.end());

    ASSERT_ALMOST_EQUAL(h_keys, d_keys);
  }
};
VariableUnitTest<TestRadixSort, RadixSortKeyTypes> TestRadixSortInstance;


template <typename T>
struct TestRadixSortVariableBits
{
  void operator()(const size_t n)
  {
    for(size_t num_bits = 0; num_bits < 8 * sizeof(T); num_bits += 7){
        thrust::host_vector<T>  h_keys = unittest::random_integers<T>(n);
   
        size_t mask = (1 << num_bits) - 1;
        for(size_t i = 0; i < n; i++)
            h_keys[i] &= mask;

        thrust::device_vector<T> d_keys = h_keys;
    
        thrust::stable_sort(h_keys.begin(), h_keys.end());
        thrust::detail::device::cuda::detail::stable_radix_sort(d_keys.begin(), d_keys.end());
    
        ASSERT_ALMOST_EQUAL(h_keys, d_keys);
    }
  }
};
VariableUnitTest<TestRadixSortVariableBits, unittest::type_list<unsigned int> > TestRadixSortVariableBitsInstance;


template <typename T>
struct TestRadixSortUnaligned
{
  void operator()(const size_t n)
  {
    typedef thrust::device_vector<T> Vector;

    Vector unsorted_keys = unittest::random_integers<T>(n);
    Vector   sorted_keys = unsorted_keys;
    
    thrust::sort(sorted_keys.begin(), sorted_keys.end());

    for(int offset = 1; offset < 16; offset++)
    {
        Vector unaligned_unsorted_keys(n + offset, 0);
        Vector   unaligned_sorted_keys(n + offset, 0);
        
        thrust::copy(unsorted_keys.begin(), unsorted_keys.end(), unaligned_unsorted_keys.begin() + offset);
        thrust::copy(  sorted_keys.begin(),   sorted_keys.end(),   unaligned_sorted_keys.begin() + offset);
   
        thrust::detail::device::cuda::detail::stable_radix_sort(unaligned_unsorted_keys.begin() + offset, unaligned_unsorted_keys.end());

        ASSERT_EQUAL(unaligned_unsorted_keys, unaligned_sorted_keys);
    }
  }
};
VariableUnitTest<TestRadixSortUnaligned, unittest::type_list<char,short,int,long> > TestRadixSortUnalignedInstance;

#endif // THRUST_DEVICE_BACKEND == THRUST_DEVICE_BACKEND_CUDA

