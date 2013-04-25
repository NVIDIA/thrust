#include <unittest/unittest.h>
#include <thrust/scatter.h>
#include <thrust/execution_policy.h>


template<typename ExecutionPolicy, typename Iterator1, typename Iterator2, typename Iterator3>
__global__
void scatter_kernel(ExecutionPolicy exec, Iterator1 first, Iterator1 last, Iterator2 map_first, Iterator3 result)
{
  thrust::scatter(exec, first, last, map_first, result);
}


template<typename T, typename ExecutionPolicy>
void TestScatterDevice(ExecutionPolicy exec, const size_t n)
{
  const size_t output_size = std::min((size_t) 10, 2 * n);
  
  thrust::host_vector<T> h_input(n, (T) 1);
  thrust::device_vector<T> d_input(n, (T) 1);
  
  thrust::host_vector<unsigned int> h_map = unittest::random_integers<unsigned int>(n);
  
  for(size_t i = 0; i < n; i++)
  {
    h_map[i] =  h_map[i] % output_size;
  }
  
  thrust::device_vector<unsigned int> d_map = h_map;
  
  thrust::host_vector<T>   h_output(output_size, (T) 0);
  thrust::device_vector<T> d_output(output_size, (T) 0);
  
  thrust::scatter(h_input.begin(), h_input.end(), h_map.begin(), h_output.begin());
  scatter_kernel<<<1,1>>>(exec, d_input.begin(), d_input.end(), d_map.begin(), d_output.begin());
  
  ASSERT_EQUAL(h_output, d_output);
}

template<typename T>
void TestScatterDeviceSeq(const size_t n)
{
  TestScatterDevice<T>(thrust::seq, n);
}
DECLARE_VARIABLE_UNITTEST(TestScatterDeviceSeq);

template<typename T>
void TestScatterDeviceDevice(const size_t n)
{
  TestScatterDevice<T>(thrust::device, n);
}
DECLARE_VARIABLE_UNITTEST(TestScatterDeviceDevice);


template<typename ExecutionPolicy, typename Iterator1, typename Iterator2, typename Iterator3, typename Iterator4, typename Function>
__global__
void scatter_if_kernel(ExecutionPolicy exec, Iterator1 first, Iterator1 last, Iterator2 map_first, Iterator3 stencil_first, Iterator4 result, Function f)
{
  thrust::scatter_if(exec, first, last, map_first, stencil_first, result, f);
}


template<typename T>
struct is_even_scatter_if
{
  __host__ __device__ bool operator()(const T i) const { return (i % 2) == 0; }
};


template<typename T, typename ExecutionPolicy>
void TestScatterIfDevice(ExecutionPolicy exec, const size_t n)
{
  const size_t output_size = std::min((size_t) 10, 2 * n);
  
  thrust::host_vector<T> h_input(n, (T) 1);
  thrust::device_vector<T> d_input(n, (T) 1);
  
  thrust::host_vector<unsigned int> h_map = unittest::random_integers<unsigned int>(n);
  
  for(size_t i = 0; i < n; i++)
  {
    h_map[i] =  h_map[i] % output_size;
  }
  
  thrust::device_vector<unsigned int> d_map = h_map;
  
  thrust::host_vector<T>   h_output(output_size, (T) 0);
  thrust::device_vector<T> d_output(output_size, (T) 0);
  
  thrust::scatter_if(h_input.begin(), h_input.end(), h_map.begin(), h_map.begin(), h_output.begin(), is_even_scatter_if<unsigned int>());
  scatter_if_kernel<<<1,1>>>(exec, d_input.begin(), d_input.end(), d_map.begin(), d_map.begin(), d_output.begin(), is_even_scatter_if<unsigned int>());
  
  ASSERT_EQUAL(h_output, d_output);
}

template<typename T>
void TestScatterIfDeviceSeq(const size_t n)
{
  TestScatterIfDevice<T>(thrust::seq, n);
}
DECLARE_VARIABLE_UNITTEST(TestScatterIfDeviceSeq);

template<typename T>
void TestScatterIfDeviceDevice(const size_t n)
{
  TestScatterIfDevice<T>(thrust::device, n);
}
DECLARE_VARIABLE_UNITTEST(TestScatterIfDeviceDevice);

