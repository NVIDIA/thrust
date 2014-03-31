#include <unittest/unittest.h>
#include <thrust/transform_reduce.h>
#include <thrust/execution_policy.h>


template<typename Iterator1, typename Function1, typename T, typename Function2, typename Iterator2>
__global__
void transform_reduce_kernel(Iterator1 first, Iterator1 last, Function1 f1, T init, Function2 f2, Iterator2 result)
{
  *result = thrust::transform_reduce(thrust::seq, first, last, f1, init, f2);
}


void TestTransformReduceDeviceSeq()
{
  typedef thrust::device_vector<int> Vector;
  typedef typename Vector::value_type T;
  
  Vector data(3);
  data[0] = 1; data[1] = -2; data[2] = 3;
  
  T init = 10;

  thrust::device_vector<T> result(1);

  transform_reduce_kernel<<<1,1>>>(data.begin(), data.end(), thrust::negate<T>(), init, thrust::plus<T>(), result.begin());
  
  ASSERT_EQUAL(8, (T)result[0]);
}
DECLARE_UNITTEST(TestTransformReduceDeviceSeq);


void TestTransformReduceCudaStreams()
{
  typedef thrust::device_vector<int> Vector;
  typedef typename Vector::value_type T;
  
  Vector data(3);
  data[0] = 1; data[1] = -2; data[2] = 3;
  
  T init = 10;

  cudaStream_t s;
  cudaStreamCreate(&s);

  T result = thrust::transform_reduce(thrust::cuda::par(s), data.begin(), data.end(), thrust::negate<T>(), init, thrust::plus<T>());
  cudaStreamSynchronize(s);
  
  ASSERT_EQUAL(8, result);

  cudaStreamDestroy(s);
}
DECLARE_UNITTEST(TestTransformReduceCudaStreams);

