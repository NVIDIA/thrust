#include <unittest/unittest.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>
#include <thrust/device_malloc_allocator.h>

#include <thrust/sort.h>
#include <thrust/detail/backend/cuda/detail/stable_radix_sort.h>

#if THRUST_DEVICE_BACKEND == THRUST_DEVICE_BACKEND_CUDA

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
struct TestRadixSortByKeyVariableBits
{
  void operator()(const size_t n)
  {
    for(size_t num_bits = 0; num_bits < 8 * sizeof(T); num_bits += 3)
    {
        thrust::host_vector<T>   h_keys = unittest::random_integers<T>(n);
   
        const T mask = (1 << num_bits) - 1;
        for(size_t i = 0; i < n; i++)
            h_keys[i] &= mask;

        thrust::device_vector<T> d_keys = h_keys;
    
        thrust::host_vector<unsigned int>   h_values(n);
        thrust::device_vector<unsigned int> d_values(n);
        thrust::sequence(h_values.begin(), h_values.end());
        thrust::sequence(d_values.begin(), d_values.end());

        thrust::stable_sort_by_key(h_keys.begin(), h_keys.end(), h_values.begin());
        thrust::detail::backend::cuda::detail::stable_radix_sort_by_key(d_keys.begin(), d_keys.end(), d_values.begin());

        ASSERT_ALMOST_EQUAL(h_keys, d_keys);
        ASSERT_ALMOST_EQUAL(h_values, d_values);
    }
  }
};
VariableUnitTest<TestRadixSortByKeyVariableBits, UnsignedIntegerTypes> TestRadixSortByKeyVariableBitsInstance;

#endif // THRUST_DEVICE_BACKEND == THRUST_DEVICE_BACKEND_CUDA

