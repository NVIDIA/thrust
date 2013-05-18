#include <unittest/unittest.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>
#include <thrust/device_malloc_allocator.h>
#include <thrust/sort.h>
#include <thrust/system/cuda/detail/detail/stable_radix_sort.h>

typedef unittest::type_list<
#if !(defined(__GNUC__) && (__GNUC__ <= 4) && (__GNUC_MINOR__ <= 1))
// XXX GCC 4.1 miscompiles the char sorts with -O2 for some reason
                            unsigned char,
#endif
                            unsigned short,
                            unsigned int,
                            unsigned long,
                            unsigned long long> UnsignedIntegerTypes;

template<typename T>
struct TestRadixSortUnaligned
{
  void operator()(const size_t n)
  {
    typedef thrust::device_vector<T> Vector;

    Vector unsorted_keys = unittest::random_integers<T>(n);
    Vector   sorted_keys = unsorted_keys;
    
    thrust::sort(sorted_keys.begin(), sorted_keys.end());

    for(int offset = 1; offset < 4; offset++)
    {
        Vector unaligned_unsorted_keys(n + offset, 0);
        Vector   unaligned_sorted_keys(n + offset, 0);
        
        thrust::copy(unsorted_keys.begin(), unsorted_keys.end(), unaligned_unsorted_keys.begin() + offset);
        thrust::copy(  sorted_keys.begin(),   sorted_keys.end(),   unaligned_sorted_keys.begin() + offset);
   
        thrust::cuda::tag cuda_tag;
        thrust::system::cuda::detail::detail::stable_radix_sort(cuda_tag, unaligned_unsorted_keys.begin() + offset, unaligned_unsorted_keys.end());

        ASSERT_EQUAL(unaligned_unsorted_keys, unaligned_sorted_keys);
    }
  }
};
VariableUnitTest<TestRadixSortUnaligned, UnsignedIntegerTypes> TestRadixSortUnalignedInstance;


template<typename ExecutionPolicy, typename Iterator1, typename Iterator2>
__global__ void stable_radix_sort_kernel(ExecutionPolicy exec, Iterator1 first, Iterator1 last, Iterator2 is_supported)
{
#if __CUDA_ARCH__ < 350
  *is_supported = false;
#else
  *is_supported = true;
  thrust::system::cuda::detail::detail::stable_radix_sort(exec, first, last);
#endif
}


template<typename T>
struct TestRadixSortUnalignedDeviceDevice
{
  void operator()(const size_t n)
  {
    typedef thrust::device_vector<T> Vector;

    Vector unsorted_keys = unittest::random_integers<T>(n);
    Vector   sorted_keys = unsorted_keys;
    
    thrust::sort(sorted_keys.begin(), sorted_keys.end());

    for(int offset = 1; offset < 4; offset++)
    {
        Vector unaligned_unsorted_keys(n + offset, 0);
        Vector   unaligned_sorted_keys(n + offset, 0);
        
        thrust::copy(unsorted_keys.begin(), unsorted_keys.end(), unaligned_unsorted_keys.begin() + offset);
        thrust::copy(  sorted_keys.begin(),   sorted_keys.end(),   unaligned_sorted_keys.begin() + offset);

        thrust::device_vector<bool> is_supported(1);
   
        stable_radix_sort_kernel<<<1,1>>>(thrust::device, unaligned_unsorted_keys.begin() + offset, unaligned_unsorted_keys.end(), is_supported.begin());

        if(is_supported[0])
        {
          ASSERT_EQUAL(unaligned_unsorted_keys, unaligned_sorted_keys);
        }
    }
  }
};
VariableUnitTest<TestRadixSortUnalignedDeviceDevice, UnsignedIntegerTypes> TestRadixSortUnalignedDeviceDeviceInstance;

