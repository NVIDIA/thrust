#include <unittest/unittest.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>


template<typename Iterator>
__global__
void sequence_kernel(Iterator first, Iterator last)
{
  thrust::sequence(thrust::seq, first, last);
}


template<typename Iterator, typename T>
__global__
void sequence_kernel(Iterator first, Iterator last, T init)
{
  thrust::sequence(thrust::seq, first, last, init);
}


template<typename Iterator, typename T>
__global__
void sequence_kernel(Iterator first, Iterator last, T init, T step)
{
  thrust::sequence(thrust::seq, first, last, init, step);
}


void TestSequenceDeviceSeq()
{
  thrust::device_vector<int> v(5);
  
  sequence_kernel<<<1,1>>>(v.begin(), v.end());
  
  ASSERT_EQUAL(v[0], 0);
  ASSERT_EQUAL(v[1], 1);
  ASSERT_EQUAL(v[2], 2);
  ASSERT_EQUAL(v[3], 3);
  ASSERT_EQUAL(v[4], 4);
  
  sequence_kernel<<<1,1>>>(v.begin(), v.end(), 10);
  
  ASSERT_EQUAL(v[0], 10);
  ASSERT_EQUAL(v[1], 11);
  ASSERT_EQUAL(v[2], 12);
  ASSERT_EQUAL(v[3], 13);
  ASSERT_EQUAL(v[4], 14);
  
  sequence_kernel<<<1,1>>>(v.begin(), v.end(), 10, 2);
  
  ASSERT_EQUAL(v[0], 10);
  ASSERT_EQUAL(v[1], 12);
  ASSERT_EQUAL(v[2], 14);
  ASSERT_EQUAL(v[3], 16);
  ASSERT_EQUAL(v[4], 18);
}
DECLARE_UNITTEST(TestSequenceDeviceSeq);


void TestSequenceCudaStreams()
{
  typedef thrust::device_vector<int> Vector;
  typedef typename Vector::value_type T;
  
  Vector v(5);

  cudaStream_t s;
  cudaStreamCreate(&s);

  thrust::sequence(thrust::cuda::par(s), v.begin(), v.end());
  cudaStreamSynchronize(s);

  ASSERT_EQUAL(v[0], 0);
  ASSERT_EQUAL(v[1], 1);
  ASSERT_EQUAL(v[2], 2);
  ASSERT_EQUAL(v[3], 3);
  ASSERT_EQUAL(v[4], 4);

  thrust::sequence(thrust::cuda::par(s), v.begin(), v.end(), 10);
  cudaStreamSynchronize(s);

  ASSERT_EQUAL(v[0], 10);
  ASSERT_EQUAL(v[1], 11);
  ASSERT_EQUAL(v[2], 12);
  ASSERT_EQUAL(v[3], 13);
  ASSERT_EQUAL(v[4], 14);
  
  thrust::sequence(thrust::cuda::par(s), v.begin(), v.end(), 10, 2);
  cudaStreamSynchronize(s);

  ASSERT_EQUAL(v[0], 10);
  ASSERT_EQUAL(v[1], 12);
  ASSERT_EQUAL(v[2], 14);
  ASSERT_EQUAL(v[3], 16);
  ASSERT_EQUAL(v[4], 18);

  cudaStreamDestroy(s);
}
DECLARE_UNITTEST(TestSequenceCudaStreams);

