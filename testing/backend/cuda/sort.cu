#include <unittest/unittest.h>
#include <thrust/sort.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>


template<typename Iterator, typename Compare, typename Iterator2>
__global__
void sort_kernel(Iterator first, Iterator last, Compare comp, Iterator2 is_supported)
{
#if (__CUDA_ARCH__ >= 200)
  *is_supported = true;
  thrust::sort(thrust::seq, first, last, comp);
#else
  *is_supported = false;
#endif
}


template<typename T>
  struct TestSortDeviceSeq
{
  void operator()(const size_t n)
  {
    thrust::host_vector<T>   h_data = unittest::random_integers<T>(n);
    thrust::device_vector<T> d_data = h_data;
    
    thrust::device_vector<bool> is_supported(1);
    sort_kernel<<<1,1>>>(d_data.begin(), d_data.end(), thrust::less<T>(), is_supported.begin());

    if(is_supported[0])
    {
      thrust::sort(h_data.begin(), h_data.end(), thrust::less<T>());
      
      ASSERT_EQUAL(h_data, d_data);
    }
  }
};
VariableUnitTest<
  TestSortDeviceSeq,
  unittest::type_list<unittest::int8_t,unittest::int32_t>
> TestSortDeviceSeqInstance;


void TestSortCudaStreams()
{
  thrust::device_vector<int> keys(10);

  keys[0] = 9;
  keys[1] = 3;
  keys[2] = 2;
  keys[3] = 0;
  keys[4] = 4;
  keys[5] = 7;
  keys[6] = 8;
  keys[7] = 1;
  keys[8] = 5;
  keys[9] = 6;

  cudaStream_t s;
  cudaStreamCreate(&s);

  thrust::sort(thrust::cuda::par(s),
               keys.begin(), keys.end());
  cudaStreamSynchronize(s);

  ASSERT_EQUAL(true, thrust::is_sorted(keys.begin(), keys.end()));
                      
  cudaStreamDestroy(s);
}
DECLARE_UNITTEST(TestSortCudaStreams);

