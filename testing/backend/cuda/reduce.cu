#include <unittest/unittest.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>


template<typename Iterator, typename T, typename Iterator2>
__global__
void reduce_kernel(Iterator first, Iterator last, T init, Iterator2 result)
{
  *result = thrust::reduce(thrust::seq, first, last, init);
}


template<typename T>
struct TestReduceDeviceSeq
{
  void operator()(const size_t n)
  {
    thrust::host_vector<T>   h_data = unittest::random_integers<T>(n);
    thrust::device_vector<T> d_data = h_data;

    thrust::device_vector<T> d_result(1);
    
    T init = 13;
    
    T h_result = thrust::reduce(h_data.begin(), h_data.end(), init);

    reduce_kernel<<<1,1>>>(d_data.begin(), d_data.end(), init, d_result.begin());
    
    ASSERT_EQUAL(h_result, d_result[0]);
  }
};
VariableUnitTest<TestReduceDeviceSeq, IntegralTypes> TestReduceDeviceSeqInstance;


void TestReduceCudaStreams()
{
  typedef thrust::device_vector<int> Vector;
  typedef typename Vector::value_type T;

  Vector v(3);
  v[0] = 1; v[1] = -2; v[2] = 3;

  cudaStream_t s;
  cudaStreamCreate(&s);

  // no initializer
  ASSERT_EQUAL(thrust::reduce(thrust::cuda::par(s), v.begin(), v.end()), 2);

  // with initializer
  ASSERT_EQUAL(thrust::reduce(thrust::cuda::par(s), v.begin(), v.end(), 10), 12);

  cudaStreamDestroy(s);
}
DECLARE_UNITTEST(TestReduceCudaStreams);

