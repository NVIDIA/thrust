#include <unittest/unittest.h>
#include <thrust/logical.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>


template<typename Iterator, typename Function, typename Iterator2>
__global__
void all_of_kernel(Iterator first, Iterator last, Function f, Iterator2 result)
{
  *result = thrust::all_of(thrust::seq, first, last, f);
}


void TestAllOfDeviceSeq()
{
  typedef int T;
  thrust::device_vector<T> v(3, 1);
  thrust::device_vector<bool> result(1);
  
  all_of_kernel<<<1,1>>>(v.begin(), v.end(), thrust::identity<T>(), result.begin());
  ASSERT_EQUAL(true, result[0]);
  
  v[1] = 0;
  
  all_of_kernel<<<1,1>>>(v.begin(), v.end(), thrust::identity<T>(), result.begin());
  ASSERT_EQUAL(false, result[0]);
  
  all_of_kernel<<<1,1>>>(v.begin() + 0, v.begin() + 0, thrust::identity<T>(), result.begin());
  ASSERT_EQUAL(true, result[0]);

  all_of_kernel<<<1,1>>>(v.begin() + 0, v.begin() + 1, thrust::identity<T>(), result.begin());
  ASSERT_EQUAL(true, result[0]);

  all_of_kernel<<<1,1>>>(v.begin() + 0, v.begin() + 2, thrust::identity<T>(), result.begin());
  ASSERT_EQUAL(false, result[0]);

  all_of_kernel<<<1,1>>>(v.begin() + 1, v.begin() + 2, thrust::identity<T>(), result.begin());
  ASSERT_EQUAL(false, result[0]);
}
DECLARE_UNITTEST(TestAllOfDeviceSeq);


void TestAllOfCudaStreams()
{
  typedef thrust::device_vector<int> Vector;
  typedef typename Vector::value_type T;
  
  Vector v(3, 1);

  cudaStream_t s;
  cudaStreamCreate(&s);
  
  ASSERT_EQUAL(thrust::all_of(thrust::cuda::par(s), v.begin(), v.end(), thrust::identity<T>()), true);
  
  v[1] = 0;
  
  ASSERT_EQUAL(thrust::all_of(thrust::cuda::par(s), v.begin(), v.end(), thrust::identity<T>()), false);
  
  ASSERT_EQUAL(thrust::all_of(thrust::cuda::par(s), v.begin() + 0, v.begin() + 0, thrust::identity<T>()), true);
  ASSERT_EQUAL(thrust::all_of(thrust::cuda::par(s), v.begin() + 0, v.begin() + 1, thrust::identity<T>()), true);
  ASSERT_EQUAL(thrust::all_of(thrust::cuda::par(s), v.begin() + 0, v.begin() + 2, thrust::identity<T>()), false);
  ASSERT_EQUAL(thrust::all_of(thrust::cuda::par(s), v.begin() + 1, v.begin() + 2, thrust::identity<T>()), false);

  cudaStreamDestroy(s);
}
DECLARE_UNITTEST(TestAllOfCudaStreams);


template<typename Iterator, typename Function, typename Iterator2>
__global__
void any_of_kernel(Iterator first, Iterator last, Function f, Iterator2 result)
{
  *result = thrust::any_of(thrust::seq, first, last, f);
}


void TestAnyOfDeviceSeq()
{
  typedef int T;
  
  thrust::device_vector<T> v(3, 1);
  thrust::device_vector<bool> result(1);
  
  any_of_kernel<<<1,1>>>(v.begin(), v.end(), thrust::identity<T>(), result.begin());
  ASSERT_EQUAL(true, result[0]);
  
  v[1] = 0;
  
  any_of_kernel<<<1,1>>>(v.begin(), v.end(), thrust::identity<T>(), result.begin());
  ASSERT_EQUAL(true, result[0]);
  
  any_of_kernel<<<1,1>>>(v.begin() + 0, v.begin() + 0, thrust::identity<T>(), result.begin());
  ASSERT_EQUAL(false, result[0]);

  any_of_kernel<<<1,1>>>(v.begin() + 0, v.begin() + 1, thrust::identity<T>(), result.begin());
  ASSERT_EQUAL(true, result[0]);

  any_of_kernel<<<1,1>>>(v.begin() + 0, v.begin() + 2, thrust::identity<T>(), result.begin());
  ASSERT_EQUAL(true, result[0]);

  any_of_kernel<<<1,1>>>(v.begin() + 1, v.begin() + 2, thrust::identity<T>(), result.begin());
  ASSERT_EQUAL(false, result[0]);
}
DECLARE_UNITTEST(TestAnyOfDeviceSeq);


void TestAnyOfCudaStreams()
{
  typedef thrust::device_vector<int> Vector;
  typedef typename Vector::value_type T;

  Vector v(3, 1);

  cudaStream_t s;
  cudaStreamCreate(&s);

  ASSERT_EQUAL(thrust::any_of(thrust::cuda::par(s), v.begin(), v.end(), thrust::identity<T>()), true);

  v[1] = 0;
  
  ASSERT_EQUAL(thrust::any_of(thrust::cuda::par(s), v.begin(), v.end(), thrust::identity<T>()), true);

  ASSERT_EQUAL(thrust::any_of(thrust::cuda::par(s), v.begin() + 0, v.begin() + 0, thrust::identity<T>()), false);
  ASSERT_EQUAL(thrust::any_of(thrust::cuda::par(s), v.begin() + 0, v.begin() + 1, thrust::identity<T>()), true);
  ASSERT_EQUAL(thrust::any_of(thrust::cuda::par(s), v.begin() + 0, v.begin() + 2, thrust::identity<T>()), true);
  ASSERT_EQUAL(thrust::any_of(thrust::cuda::par(s), v.begin() + 1, v.begin() + 2, thrust::identity<T>()), false);

  cudaStreamDestroy(s);
}
DECLARE_UNITTEST(TestAnyOfCudaStreams);


template<typename Iterator, typename Function, typename Iterator2>
__global__
void none_of_kernel(Iterator first, Iterator last, Function f, Iterator2 result)
{
  *result = thrust::none_of(thrust::seq, first, last, f);
}


void TestNoneOfDeviceSeq()
{
  typedef int T;
  
  thrust::device_vector<T> v(3, 1);
  thrust::device_vector<bool> result(1);
  
  none_of_kernel<<<1,1>>>(v.begin(), v.end(), thrust::identity<T>(), result.begin());
  ASSERT_EQUAL(false, result[0]);
  
  v[1] = 0;
  
  none_of_kernel<<<1,1>>>(v.begin(), v.end(), thrust::identity<T>(), result.begin());
  ASSERT_EQUAL(false, result[0]);
  
  none_of_kernel<<<1,1>>>(v.begin() + 0, v.begin() + 0, thrust::identity<T>(), result.begin());
  ASSERT_EQUAL(true, result[0]);

  none_of_kernel<<<1,1>>>(v.begin() + 0, v.begin() + 1, thrust::identity<T>(), result.begin());
  ASSERT_EQUAL(false, result[0]);

  none_of_kernel<<<1,1>>>(v.begin() + 0, v.begin() + 2, thrust::identity<T>(), result.begin());
  ASSERT_EQUAL(false, result[0]);

  none_of_kernel<<<1,1>>>(v.begin() + 1, v.begin() + 2, thrust::identity<T>(), result.begin());
  ASSERT_EQUAL(true, result[0]);
}
DECLARE_UNITTEST(TestNoneOfDeviceSeq);


void TestNoneOfCudaStreams()
{
  typedef thrust::device_vector<int> Vector;
  typedef typename Vector::value_type T;

  Vector v(3, 1);

  cudaStream_t s;
  cudaStreamCreate(&s);

  ASSERT_EQUAL(thrust::none_of(thrust::cuda::par(s), v.begin(), v.end(), thrust::identity<T>()), false);

  v[1] = 0;
  
  ASSERT_EQUAL(thrust::none_of(thrust::cuda::par(s), v.begin(), v.end(), thrust::identity<T>()), false);

  ASSERT_EQUAL(thrust::none_of(thrust::cuda::par(s), v.begin() + 0, v.begin() + 0, thrust::identity<T>()), true);
  ASSERT_EQUAL(thrust::none_of(thrust::cuda::par(s), v.begin() + 0, v.begin() + 1, thrust::identity<T>()), false);
  ASSERT_EQUAL(thrust::none_of(thrust::cuda::par(s), v.begin() + 0, v.begin() + 2, thrust::identity<T>()), false);
  ASSERT_EQUAL(thrust::none_of(thrust::cuda::par(s), v.begin() + 1, v.begin() + 2, thrust::identity<T>()), true);

  cudaStreamDestroy(s);
}
DECLARE_UNITTEST(TestNoneOfCudaStreams);

