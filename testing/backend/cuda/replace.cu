#include <unittest/unittest.h>
#include <thrust/replace.h>
#include <thrust/execution_policy.h>


template <typename T>
struct less_than_five
{
  __host__ __device__ bool operator()(const T &val) const {return val < 5;}
};


template<typename Iterator, typename T1, typename T2>
__global__
void replace_kernel(Iterator first, Iterator last, T1 old_value, T2 new_value)
{
  thrust::replace(thrust::seq, first, last, old_value, new_value);
}


template<typename T>
void TestReplaceDeviceSeq(const size_t n)
{
  thrust::host_vector<T>   h_data = unittest::random_samples<T>(n);
  thrust::device_vector<T> d_data = h_data;
  
  T old_value = 0;
  T new_value = 1;
  
  thrust::replace(thrust::seq, h_data.begin(), h_data.end(), old_value, new_value);
  replace_kernel<<<1,1>>>(d_data.begin(), d_data.end(), old_value, new_value);
  
  ASSERT_ALMOST_EQUAL(h_data, d_data);
}
DECLARE_VARIABLE_UNITTEST(TestReplaceDeviceSeq);


template<typename Iterator1, typename Iterator2, typename T1, typename T2>
__global__
void replace_copy_kernel(Iterator1 first, Iterator1 last, Iterator2 result, T1 old_value, T2 new_value)
{
  thrust::replace_copy(thrust::seq, first, last, result, old_value, new_value);
}


template<typename T>
void TestReplaceCopyDeviceSeq(const size_t n)
{
  thrust::host_vector<T>   h_data = unittest::random_samples<T>(n);
  thrust::device_vector<T> d_data = h_data;
  
  T old_value = 0;
  T new_value = 1;
  
  thrust::host_vector<T>   h_dest(n);
  thrust::device_vector<T> d_dest(n);
  
  thrust::replace_copy(h_data.begin(), h_data.end(), h_dest.begin(), old_value, new_value);
  replace_copy_kernel<<<1,1>>>(d_data.begin(), d_data.end(), d_dest.begin(), old_value, new_value);
  
  ASSERT_ALMOST_EQUAL(h_data, d_data);
  ASSERT_ALMOST_EQUAL(h_dest, d_dest);
}
DECLARE_VARIABLE_UNITTEST(TestReplaceCopyDeviceSeq);


template<typename Iterator, typename Predicate, typename T>
__global__
void replace_if_kernel(Iterator first, Iterator last, Predicate pred, T new_value)
{
  thrust::replace_if(thrust::seq, first, last, pred, new_value);
}


template<typename T>
void TestReplaceIfDeviceSeq(const size_t n)
{
  thrust::host_vector<T>   h_data = unittest::random_samples<T>(n);
  thrust::device_vector<T> d_data = h_data;
  
  thrust::replace_if(h_data.begin(), h_data.end(), less_than_five<T>(), (T) 0);
  replace_if_kernel<<<1,1>>>(d_data.begin(), d_data.end(), less_than_five<T>(), (T) 0);
  
  ASSERT_ALMOST_EQUAL(h_data, d_data);
}
DECLARE_VARIABLE_UNITTEST(TestReplaceIfDeviceSeq);


template<typename Iterator1, typename Iterator2, typename Predicate, typename T>
__global__
void replace_if_kernel(Iterator1 first, Iterator1 last, Iterator2 stencil_first, Predicate pred, T new_value)
{
  thrust::replace_if(thrust::seq, first, last, stencil_first, pred, new_value);
}


template<typename T>
void TestReplaceIfStencilDeviceSeq(const size_t n)
{
  thrust::host_vector<T>   h_data = unittest::random_samples<T>(n);
  thrust::device_vector<T> d_data = h_data;
  
  thrust::host_vector<T>   h_stencil = unittest::random_samples<T>(n);
  thrust::device_vector<T> d_stencil = h_stencil;
  
  thrust::replace_if(h_data.begin(), h_data.end(), h_stencil.begin(), less_than_five<T>(), (T) 0);
  replace_if_kernel<<<1,1>>>(d_data.begin(), d_data.end(), d_stencil.begin(), less_than_five<T>(), (T) 0);
  
  ASSERT_ALMOST_EQUAL(h_data, d_data);
}
DECLARE_VARIABLE_UNITTEST(TestReplaceIfStencilDeviceSeq);


template<typename Iterator1, typename Iterator2, typename Predicate, typename T>
__global__
void replace_copy_if_kernel(Iterator1 first, Iterator1 last, Iterator2 result, Predicate pred, T new_value)
{
  thrust::replace_copy_if(thrust::seq, first, last, result, pred, new_value);
}


template<typename T>
void TestReplaceCopyIfDeviceSeq(const size_t n)
{
  thrust::host_vector<T>   h_data = unittest::random_samples<T>(n);
  thrust::device_vector<T> d_data = h_data;
  
  thrust::host_vector<T>   h_dest(n);
  thrust::device_vector<T> d_dest(n);
  
  thrust::replace_copy_if(h_data.begin(), h_data.end(), h_dest.begin(), less_than_five<T>(), 0);
  replace_copy_if_kernel<<<1,1>>>(d_data.begin(), d_data.end(), d_dest.begin(), less_than_five<T>(), 0);
  
  ASSERT_ALMOST_EQUAL(h_data, d_data);
  ASSERT_ALMOST_EQUAL(h_dest, d_dest);
}
DECLARE_VARIABLE_UNITTEST(TestReplaceCopyIfDeviceSeq);


template<typename Iterator1, typename Iterator2, typename Iterator3, typename Predicate, typename T>
__global__
void replace_copy_if_kernel(Iterator1 first, Iterator1 last, Iterator2 stencil_first, Iterator3 result, Predicate pred, T new_value)
{
  thrust::replace_copy_if(thrust::seq, first, last, stencil_first, result, pred, new_value);
}


template<typename T>
void TestReplaceCopyIfStencilDeviceSeq(const size_t n)
{
  thrust::host_vector<T>   h_data = unittest::random_samples<T>(n);
  thrust::device_vector<T> d_data = h_data;
  
  thrust::host_vector<T>   h_stencil = unittest::random_samples<T>(n);
  thrust::device_vector<T> d_stencil = h_stencil;
  
  thrust::host_vector<T>   h_dest(n);
  thrust::device_vector<T> d_dest(n);
  
  thrust::replace_copy_if(h_data.begin(), h_data.end(), h_stencil.begin(), h_dest.begin(), less_than_five<T>(), 0);
  replace_copy_if_kernel<<<1,1>>>(d_data.begin(), d_data.end(), d_stencil.begin(), d_dest.begin(), less_than_five<T>(), 0);
  
  ASSERT_ALMOST_EQUAL(h_data, d_data);
  ASSERT_ALMOST_EQUAL(h_dest, d_dest);
}
DECLARE_VARIABLE_UNITTEST(TestReplaceCopyIfStencilDeviceSeq);


void TestReplaceCudaStreams()
{
  typedef thrust::device_vector<int> Vector;
  typedef typename Vector::value_type T;

  Vector data(5);
  data[0] =  1; 
  data[1] =  2; 
  data[2] =  1;
  data[3] =  3; 
  data[4] =  2; 

  cudaStream_t s;
  cudaStreamCreate(&s);

  thrust::replace(thrust::cuda::par(s), data.begin(), data.end(), (T) 1, (T) 4);
  thrust::replace(thrust::cuda::par(s), data.begin(), data.end(), (T) 2, (T) 5);

  cudaStreamSynchronize(s);

  Vector result(5);
  result[0] =  4; 
  result[1] =  5; 
  result[2] =  4;
  result[3] =  3; 
  result[4] =  5; 

  ASSERT_EQUAL(data, result);

  cudaStreamDestroy(s);
}
DECLARE_UNITTEST(TestReplaceCudaStreams);

