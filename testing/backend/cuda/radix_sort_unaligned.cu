#include <unittest/unittest.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>
#include <thrust/device_malloc_allocator.h>

#include <thrust/sort.h>
#include <thrust/system/cuda/detail/detail/stable_radix_sort.h>

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA

typedef unittest::type_list<
#if !(defined(__GNUC__) && (__GNUC__ <= 4) && (__GNUC_MINOR__ <= 1))
// XXX GCC 4.1 miscompiles the char sorts with -O2 for some reason
                            unsigned char,
#endif
                            unsigned short,
                            unsigned int,
                            unsigned long,
                            unsigned long long> UnsignedIntegerTypes;

template <typename T>
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

#endif // THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA

