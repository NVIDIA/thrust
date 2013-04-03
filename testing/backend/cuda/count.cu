#include <unittest/unittest.h>
#include <thrust/count.h>
#include <thrust/execution_policy.h>


template<typename Iterator, typename T, typename Iterator2>
__global__
void count_kernel(Iterator first, Iterator last, T value, Iterator2 result)
{
  *result = thrust::count(thrust::seq, first, last, value);
}


template<typename T>
void TestCountDeviceSeq(const size_t n)
{
  thrust::host_vector<T>   h_data = unittest::random_samples<T>(n);
  thrust::device_vector<T> d_data = h_data;

  thrust::device_vector<size_t> d_result(1);
  
  size_t h_result = thrust::count(h_data.begin(), h_data.end(), T(5));

  count_kernel<<<1,1>>>(d_data.begin(), d_data.end(), T(5), d_result.begin());
  
  ASSERT_EQUAL(h_result, d_result[0]);
}
DECLARE_VARIABLE_UNITTEST(TestCountDeviceSeq);


template<typename Iterator, typename Predicate, typename Iterator2>
__global__
void count_if_kernel(Iterator first, Iterator last, Predicate pred, Iterator2 result)
{
  *result = thrust::count_if(thrust::seq, first, last, pred);
}


template<typename T>
struct greater_than_five
{
  __host__ __device__ bool operator()(const T &x) const {return x > 5;}
};


template<typename T>
void TestCountIfDeviceSeq(const size_t n)
{
  thrust::host_vector<T>   h_data = unittest::random_samples<T>(n);
  thrust::device_vector<T> d_data = h_data;

  thrust::device_vector<size_t> d_result(1);
  
  size_t h_result = thrust::count_if(h_data.begin(), h_data.end(), greater_than_five<T>());
  count_if_kernel<<<1,1>>>(d_data.begin(), d_data.end(), greater_than_five<T>(), d_result.begin());
  
  ASSERT_EQUAL(h_result, d_result[0]);
}
DECLARE_VARIABLE_UNITTEST(TestCountIfDeviceSeq);

