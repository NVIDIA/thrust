#include <unittest/unittest.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>


template<typename Iterator1, typename Iterator2, typename Compare, typename Iterator3>
__global__
void sort_by_key_kernel(Iterator1 keys_first, Iterator1 keys_last, Iterator2 values_first, Compare comp, Iterator3 is_supported)
{
#if (__CUDA_ARCH__ >= 200)
  *is_supported = true;
  thrust::sort_by_key(thrust::seq, keys_first, keys_last, values_first, comp);
#else
  *is_supported = false;
#endif
}


template<typename T>
  struct TestSortByKeyDeviceSeq
{
  void operator()(const size_t n)
  {
    thrust::host_vector<T>   h_keys = unittest::random_integers<T>(n);
    thrust::device_vector<T> d_keys = h_keys;
    
    thrust::host_vector<T>   h_values = h_keys;
    thrust::device_vector<T> d_values = d_keys;
    
    thrust::device_vector<bool> is_supported(1);
    sort_by_key_kernel<<<1,1>>>(d_keys.begin(), d_keys.end(), d_values.begin(), thrust::less<T>(), is_supported.begin());

    if(is_supported[0])
    {
      thrust::sort_by_key(h_keys.begin(), h_keys.end(), h_values.begin(), thrust::less<T>());
      
      ASSERT_EQUAL(h_keys,   d_keys);
      ASSERT_EQUAL(h_values, d_values);
    }
  }
};
VariableUnitTest<
  TestSortByKeyDeviceSeq,
  unittest::type_list<unittest::int8_t,unittest::int32_t>
> TestSortByKeyDeviceSeqInstance;

