#include <unittest/unittest.h>
#include <thrust/inner_product.h>
#include <thrust/execution_policy.h>


template<typename Iterator1, typename Iterator2, typename T, typename Iterator3>
__global__
void inner_product_kernel(Iterator1 first1, Iterator1 last1, Iterator2 first2, T init, Iterator3 result)
{
  *result = thrust::inner_product(thrust::seq, first1, last1, first2, init);
}


template<typename T>
struct TestInnerProductDeviceSeq
{
  void operator()(const size_t n)
  {
    thrust::host_vector<T> h_v1 = unittest::random_integers<T>(n);
    thrust::host_vector<T> h_v2 = unittest::random_integers<T>(n);
    
    thrust::device_vector<T> d_v1 = h_v1;
    thrust::device_vector<T> d_v2 = h_v2;

    thrust::device_vector<T> result(1);
    
    T init = 13;
    
    T expected = thrust::inner_product(h_v1.begin(), h_v1.end(), h_v2.begin(), init);
    inner_product_kernel<<<1,1>>>(d_v1.begin(), d_v1.end(), d_v2.begin(), init, result.begin());
    
    ASSERT_EQUAL(expected, result[0]);
  }
};
VariableUnitTest<TestInnerProductDeviceSeq, IntegralTypes> TestInnerProductDeviceSeqInstance;


void TestInnerProductCudaStreams()
{
  thrust::device_vector<int> v1(3);
  thrust::device_vector<int> v2(3);
  v1[0] =  1; v1[1] = -2; v1[2] =  3;
  v2[0] = -4; v2[1] =  5; v2[2] =  6;

  cudaStream_t s;
  cudaStreamCreate(&s);
  
  int init = 3;
  int result = thrust::inner_product(thrust::cuda::par(s), v1.begin(), v1.end(), v2.begin(), init);
  ASSERT_EQUAL(result, 7);

  cudaStreamDestroy(s);
}
DECLARE_UNITTEST(TestInnerProductCudaStreams);

