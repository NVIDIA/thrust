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
struct TestRadixSortByKeyShortValues
{
  void operator()(const size_t n)
  {
    thrust::host_vector<T>   h_keys = unittest::random_integers<T>(n);
    thrust::device_vector<T> d_keys = h_keys;
    
    thrust::host_vector<short>   h_values(n);
    thrust::device_vector<short> d_values(n);
    thrust::sequence(h_values.begin(), h_values.end());
    thrust::sequence(d_values.begin(), d_values.end());

    thrust::stable_sort_by_key(h_keys.begin(), h_keys.end(), h_values.begin());

    thrust::cuda::tag cuda_tag;
    thrust::system::cuda::detail::detail::stable_radix_sort_by_key(cuda_tag, d_keys.begin(), d_keys.end(), d_values.begin(), thrust::less<T>());

    ASSERT_ALMOST_EQUAL(h_keys, d_keys);
    ASSERT_ALMOST_EQUAL(h_values, d_values);
  }
};
VariableUnitTest<TestRadixSortByKeyShortValues, UnsignedIntegerTypes> TestRadixSortByKeyShortValuesInstance;

template <typename T>
struct TestRadixSortByKeyLongLongValues
{
  void operator()(const size_t n)
  {
    thrust::host_vector<T>   h_keys = unittest::random_integers<T>(n);
    thrust::device_vector<T> d_keys = h_keys;
    
    thrust::host_vector<long long>   h_values(n);
    thrust::device_vector<long long> d_values(n);
    thrust::sequence(h_values.begin(), h_values.end());
    thrust::sequence(d_values.begin(), d_values.end());

    thrust::stable_sort_by_key(h_keys.begin(), h_keys.end(), h_values.begin());

    thrust::cuda::tag cuda_tag;
    thrust::system::cuda::detail::detail::stable_radix_sort_by_key(cuda_tag, d_keys.begin(), d_keys.end(), d_values.begin(), thrust::less<T>());

    ASSERT_ALMOST_EQUAL(h_keys, d_keys);
    ASSERT_ALMOST_EQUAL(h_values, d_values);
  }
};
VariableUnitTest<TestRadixSortByKeyLongLongValues, UnsignedIntegerTypes> TestRadixSortByKeyLongLongValuesInstance;

#endif // THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA

