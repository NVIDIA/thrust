#include <unittest/unittest.h>
#include <thrust/uninitialized_copy.h>
#include <thrust/execution_policy.h>


template<typename Iterator1, typename Iterator2>
__global__
void uninitialized_copy_kernel(Iterator1 first, Iterator1 last, Iterator2 result)
{
  thrust::uninitialized_copy(thrust::seq, first, last, result);
}


void TestUninitializedCopyDeviceSeq()
{
  typedef thrust::device_vector<int> Vector;
  typedef typename Vector::value_type T;
  
  Vector v1(5);
  v1[0] = 0; v1[1] = 1; v1[2] = 2; v1[3] = 3; v1[4] = 4;
  
  // copy to Vector
  Vector v2(5);
  uninitialized_copy_kernel<<<1,1>>>(v1.begin(), v1.end(), v2.begin());
  ASSERT_EQUAL(v2[0], 0);
  ASSERT_EQUAL(v2[1], 1);
  ASSERT_EQUAL(v2[2], 2);
  ASSERT_EQUAL(v2[3], 3);
  ASSERT_EQUAL(v2[4], 4);
}
DECLARE_UNITTEST(TestUninitializedCopyDeviceSeq);


void TestUninitializedCopyCudaStreams()
{
  typedef thrust::device_vector<int> Vector;
  typedef typename Vector::value_type T;
  
  Vector v1(5);
  v1[0] = 0; v1[1] = 1; v1[2] = 2; v1[3] = 3; v1[4] = 4;
  
  // copy to Vector
  Vector v2(5);

  cudaStream_t s;
  cudaStreamCreate(&s);

  thrust::uninitialized_copy(thrust::cuda::par(s), v1.begin(), v1.end(), v2.begin());
  cudaStreamSynchronize(s);

  ASSERT_EQUAL(v2[0], 0);
  ASSERT_EQUAL(v2[1], 1);
  ASSERT_EQUAL(v2[2], 2);
  ASSERT_EQUAL(v2[3], 3);
  ASSERT_EQUAL(v2[4], 4);

  cudaStreamDestroy(s);
}
DECLARE_UNITTEST(TestUninitializedCopyCudaStreams);


template<typename Iterator1, typename Size, typename Iterator2>
__global__
void uninitialized_copy_n_kernel(Iterator1 first, Size n, Iterator2 result)
{
  thrust::uninitialized_copy_n(thrust::seq, first, n, result);
}


void TestUninitializedCopyNDeviceSeq()
{
  typedef thrust::device_vector<int> Vector;
  typedef typename Vector::value_type T;
  
  Vector v1(5);
  v1[0] = 0; v1[1] = 1; v1[2] = 2; v1[3] = 3; v1[4] = 4;
  
  // copy to Vector
  Vector v2(5);
  uninitialized_copy_n_kernel<<<1,1>>>(v1.begin(), v1.size(), v2.begin());
  ASSERT_EQUAL(v2[0], 0);
  ASSERT_EQUAL(v2[1], 1);
  ASSERT_EQUAL(v2[2], 2);
  ASSERT_EQUAL(v2[3], 3);
  ASSERT_EQUAL(v2[4], 4);
}
DECLARE_UNITTEST(TestUninitializedCopyNDeviceSeq);


void TestUninitializedCopyNCudaStreams()
{
  typedef thrust::device_vector<int> Vector;
  typedef typename Vector::value_type T;
  
  Vector v1(5);
  v1[0] = 0; v1[1] = 1; v1[2] = 2; v1[3] = 3; v1[4] = 4;
  
  // copy to Vector
  Vector v2(5);

  cudaStream_t s;
  cudaStreamCreate(&s);

  thrust::uninitialized_copy_n(thrust::cuda::par(s), v1.begin(), v1.size(), v2.begin());
  cudaStreamSynchronize(s);

  ASSERT_EQUAL(v2[0], 0);
  ASSERT_EQUAL(v2[1], 1);
  ASSERT_EQUAL(v2[2], 2);
  ASSERT_EQUAL(v2[3], 3);
  ASSERT_EQUAL(v2[4], 4);

  cudaStreamDestroy(s);
}
DECLARE_UNITTEST(TestUninitializedCopyNCudaStreams);

