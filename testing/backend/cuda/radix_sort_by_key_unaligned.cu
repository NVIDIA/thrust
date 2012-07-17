#include <unittest/unittest.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>
#include <thrust/device_malloc_allocator.h>

#include <thrust/sort.h>
#include <thrust/system/cuda/detail/detail/stable_radix_sort.h>

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA

using namespace unittest;

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
struct TestRadixSortByKeyUnaligned
{
  void operator()(const size_t n)
  {
    typedef thrust::device_vector<T>   Vector1;
    typedef thrust::device_vector<int> Vector2;

    Vector1 unsorted_keys = unittest::random_integers<T>(n);
    Vector1   sorted_keys = unsorted_keys;

    Vector2 unsorted_values(n); thrust::sequence(unsorted_values.begin(), unsorted_values.end());
    Vector2   sorted_values = unsorted_values;
    
    thrust::sort_by_key(sorted_keys.begin(), sorted_keys.end(), sorted_values.begin());

    for(int offset = 1; offset < 4; offset++)
    {
        Vector1   unaligned_unsorted_keys(n + offset, 0);
        Vector1     unaligned_sorted_keys(n + offset, 0);
        Vector2 unaligned_unsorted_values(n + offset, 0);
        Vector2   unaligned_sorted_values(n + offset, 0);
        
        thrust::copy(  unsorted_keys.begin(),   unsorted_keys.end(),   unaligned_unsorted_keys.begin() + offset);
        thrust::copy(    sorted_keys.begin(),     sorted_keys.end(),     unaligned_sorted_keys.begin() + offset);
        thrust::copy(unsorted_values.begin(), unsorted_values.end(), unaligned_unsorted_values.begin() + offset);
        thrust::copy(  sorted_values.begin(),   sorted_values.end(),   unaligned_sorted_values.begin() + offset);
   
        thrust::cuda::tag cuda_tag;
        thrust::system::cuda::detail::detail::stable_radix_sort_by_key(cuda_tag, unaligned_unsorted_keys.begin() + offset, unaligned_unsorted_keys.end(), unaligned_unsorted_values.begin() + offset);

        ASSERT_EQUAL(  unaligned_unsorted_keys,   unaligned_sorted_keys);
        ASSERT_EQUAL(unaligned_unsorted_values, unaligned_sorted_values);
    }
  }
};
VariableUnitTest<TestRadixSortByKeyUnaligned, UnsignedIntegerTypes> TestRadixSortByKeyUnalignedInstance;

#endif // THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA

