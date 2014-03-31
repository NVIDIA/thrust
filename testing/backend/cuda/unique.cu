#include <unittest/unittest.h>
#include <thrust/unique.h>
#include <thrust/execution_policy.h>


template<typename Iterator1, typename Iterator2>
__global__
void unique_kernel(Iterator1 first, Iterator1 last, Iterator2 result)
{
  *result = thrust::unique(thrust::seq, first, last);
}


template<typename Iterator1, typename BinaryPredicate, typename Iterator2>
__global__
void unique_kernel(Iterator1 first, Iterator1 last, BinaryPredicate pred, Iterator2 result)
{
  *result = thrust::unique(thrust::seq, first, last, pred);
}


template<typename T>
struct is_equal_div_10_unique
{
  __host__ __device__
  bool operator()(const T x, const T& y) const { return ((int) x / 10) == ((int) y / 10); }
};


void TestUniqueDeviceSeq()
{
  typedef thrust::device_vector<int> Vector;
  typedef Vector::value_type T;

  Vector data(10);
  data[0] = 11; 
  data[1] = 11; 
  data[2] = 12;
  data[3] = 20; 
  data[4] = 29; 
  data[5] = 21; 
  data[6] = 21; 
  data[7] = 31; 
  data[8] = 31; 
  data[9] = 37; 

  thrust::device_vector<Vector::iterator> new_last_vec(1);
  Vector::iterator new_last;
  
  unique_kernel<<<1,1>>>(data.begin(), data.end(), new_last_vec.begin());
  new_last = new_last_vec[0];

  ASSERT_EQUAL(new_last - data.begin(), 7);
  ASSERT_EQUAL(data[0], 11);
  ASSERT_EQUAL(data[1], 12);
  ASSERT_EQUAL(data[2], 20);
  ASSERT_EQUAL(data[3], 29);
  ASSERT_EQUAL(data[4], 21);
  ASSERT_EQUAL(data[5], 31);
  ASSERT_EQUAL(data[6], 37);

  unique_kernel<<<1,1>>>(data.begin(), new_last, is_equal_div_10_unique<T>(), new_last_vec.begin());
  new_last = new_last_vec[0];

  ASSERT_EQUAL(new_last - data.begin(), 3);
  ASSERT_EQUAL(data[0], 11);
  ASSERT_EQUAL(data[1], 20);
  ASSERT_EQUAL(data[2], 31);
}
DECLARE_UNITTEST(TestUniqueDeviceSeq);


void TestUniqueCudaStreams()
{
  typedef thrust::device_vector<int> Vector;
  typedef Vector::value_type T;

  Vector data(10);
  data[0] = 11; 
  data[1] = 11; 
  data[2] = 12;
  data[3] = 20; 
  data[4] = 29; 
  data[5] = 21; 
  data[6] = 21; 
  data[7] = 31; 
  data[8] = 31; 
  data[9] = 37; 

  thrust::device_vector<Vector::iterator> new_last_vec(1);
  Vector::iterator new_last;

  cudaStream_t s;
  cudaStreamCreate(&s);
  
  new_last = thrust::unique(thrust::cuda::par(s), data.begin(), data.end());
  cudaStreamSynchronize(s);

  ASSERT_EQUAL(new_last - data.begin(), 7);
  ASSERT_EQUAL(data[0], 11);
  ASSERT_EQUAL(data[1], 12);
  ASSERT_EQUAL(data[2], 20);
  ASSERT_EQUAL(data[3], 29);
  ASSERT_EQUAL(data[4], 21);
  ASSERT_EQUAL(data[5], 31);
  ASSERT_EQUAL(data[6], 37);

  new_last = thrust::unique(thrust::cuda::par(s), data.begin(), new_last, is_equal_div_10_unique<T>());
  cudaStreamSynchronize(s);

  ASSERT_EQUAL(new_last - data.begin(), 3);
  ASSERT_EQUAL(data[0], 11);
  ASSERT_EQUAL(data[1], 20);
  ASSERT_EQUAL(data[2], 31);

  cudaStreamDestroy(s);
}
DECLARE_UNITTEST(TestUniqueCudaStreams);


template<typename Iterator1, typename Iterator2, typename Iterator3>
__global__
void unique_copy_kernel(Iterator1 first, Iterator1 last, Iterator2 result1, Iterator3 result2)
{
  *result2 = thrust::unique_copy(thrust::seq, first, last, result1);
}


template<typename Iterator1, typename Iterator2, typename BinaryPredicate, typename Iterator3>
__global__
void unique_copy_kernel(Iterator1 first, Iterator1 last, Iterator2 result1, BinaryPredicate pred, Iterator3 result2)
{
  *result2 = thrust::unique_copy(thrust::seq, first, last, result1, pred);
}


void TestUniqueCopyDeviceSeq()
{
  typedef thrust::device_vector<int> Vector;
  typedef Vector::value_type T;

  Vector data(10);
  data[0] = 11; 
  data[1] = 11; 
  data[2] = 12;
  data[3] = 20; 
  data[4] = 29; 
  data[5] = 21; 
  data[6] = 21; 
  data[7] = 31; 
  data[8] = 31; 
  data[9] = 37; 
  
  Vector output(10, -1);

  thrust::device_vector<Vector::iterator> new_last_vec(1);
  Vector::iterator new_last;
  
  unique_copy_kernel<<<1,1>>>(data.begin(), data.end(), output.begin(), new_last_vec.begin());
  new_last = new_last_vec[0];

  ASSERT_EQUAL(new_last - output.begin(), 7);
  ASSERT_EQUAL(output[0], 11);
  ASSERT_EQUAL(output[1], 12);
  ASSERT_EQUAL(output[2], 20);
  ASSERT_EQUAL(output[3], 29);
  ASSERT_EQUAL(output[4], 21);
  ASSERT_EQUAL(output[5], 31);
  ASSERT_EQUAL(output[6], 37);

  unique_copy_kernel<<<1,1>>>(output.begin(), new_last, data.begin(), is_equal_div_10_unique<T>(), new_last_vec.begin());
  new_last = new_last_vec[0];

  ASSERT_EQUAL(new_last - data.begin(), 3);
  ASSERT_EQUAL(data[0], 11);
  ASSERT_EQUAL(data[1], 20);
  ASSERT_EQUAL(data[2], 31);
}
DECLARE_UNITTEST(TestUniqueCopyDeviceSeq);


void TestUniqueCopyCudaStreams()
{
  typedef thrust::device_vector<int> Vector;
  typedef Vector::value_type T;

  Vector data(10);
  data[0] = 11; 
  data[1] = 11; 
  data[2] = 12;
  data[3] = 20; 
  data[4] = 29; 
  data[5] = 21; 
  data[6] = 21; 
  data[7] = 31; 
  data[8] = 31; 
  data[9] = 37; 
  
  Vector output(10, -1);

  thrust::device_vector<Vector::iterator> new_last_vec(1);
  Vector::iterator new_last;

  cudaStream_t s;
  cudaStreamCreate(&s);
  
  new_last = thrust::unique_copy(thrust::cuda::par(s), data.begin(), data.end(), output.begin());
  cudaStreamSynchronize(s);

  ASSERT_EQUAL(new_last - output.begin(), 7);
  ASSERT_EQUAL(output[0], 11);
  ASSERT_EQUAL(output[1], 12);
  ASSERT_EQUAL(output[2], 20);
  ASSERT_EQUAL(output[3], 29);
  ASSERT_EQUAL(output[4], 21);
  ASSERT_EQUAL(output[5], 31);
  ASSERT_EQUAL(output[6], 37);

  new_last = thrust::unique_copy(thrust::cuda::par(s), output.begin(), new_last, data.begin(), is_equal_div_10_unique<T>());
  cudaStreamSynchronize(s);

  ASSERT_EQUAL(new_last - data.begin(), 3);
  ASSERT_EQUAL(data[0], 11);
  ASSERT_EQUAL(data[1], 20);
  ASSERT_EQUAL(data[2], 31);

  cudaStreamDestroy(s);
}
DECLARE_UNITTEST(TestUniqueCopyCudaStreams);

