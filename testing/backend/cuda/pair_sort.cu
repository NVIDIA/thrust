#include <unittest/unittest.h>
#include <thrust/pair.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>


template<typename Iterator1, typename Iterator2>
__global__
void stable_sort_kernel(Iterator1 first, Iterator1 last, Iterator2 is_supported)
{
#if (__CUDA_ARCH__ >= 200)
  *is_supported = true;
  thrust::stable_sort(thrust::seq, first, last);
#else
  *is_supported = false;
#endif
}


struct make_pair_functor
{
  template<typename T1, typename T2>
  __host__ __device__
    thrust::pair<T1,T2> operator()(const T1 &x, const T2 &y)
  {
    return thrust::make_pair(x,y);
  } // end operator()()
}; // end make_pair_functor


template<typename T>
  struct TestPairStableSortDeviceSeq
{
  void operator()(const size_t n)
  {
    typedef thrust::pair<T,T> P;

    thrust::host_vector<T>   h_p1 = unittest::random_integers<T>(n);
    thrust::host_vector<T>   h_p2 = unittest::random_integers<T>(n);
    thrust::host_vector<P>   h_pairs(n);

    // zip up pairs on the host
    thrust::transform(h_p1.begin(), h_p1.end(), h_p2.begin(), h_pairs.begin(), make_pair_functor());

    thrust::device_vector<P> d_pairs = h_pairs;

    thrust::device_vector<bool> is_supported(1);

    if(is_supported[0])
    {
      // sort on the device
      thrust::stable_sort(d_pairs.begin(), d_pairs.end());

      // sort on the host
      thrust::stable_sort(h_pairs.begin(), h_pairs.end());

      ASSERT_EQUAL_QUIET(h_pairs, d_pairs);
    }
  }
};
VariableUnitTest<
  TestPairStableSortDeviceSeq,
  unittest::type_list<unittest::int8_t,unittest::int32_t>
> TestPairStableSortInstance;

