#include <unittest/unittest.h>
#include <thrust/gather.h>
#include <thrust/execution_policy.h>


template<typename Iterator1, typename Iterator2, typename Iterator3>
__global__
void gather_kernel(Iterator1 map_first, Iterator1 map_last, Iterator2 elements_first, Iterator3 result)
{
  thrust::gather(thrust::seq, map_first, map_last, elements_first, result);
}


template<typename T>
void TestGatherDeviceSeq(const size_t n)
{
  const size_t source_size = std::min((size_t) 10, 2 * n);
  
  // source vectors to gather from
  thrust::host_vector<T>   h_source = unittest::random_samples<T>(source_size);
  thrust::device_vector<T> d_source = h_source;
  
  // gather indices
  thrust::host_vector<unsigned int> h_map = unittest::random_integers<unsigned int>(n);
  
  for(size_t i = 0; i < n; i++)
    h_map[i] =  h_map[i] % source_size;
  
  thrust::device_vector<unsigned int> d_map = h_map;
  
  // gather destination
  thrust::host_vector<T>   h_output(n);
  thrust::device_vector<T> d_output(n);
  
  thrust::gather(h_map.begin(), h_map.end(), h_source.begin(), h_output.begin());
  gather_kernel<<<1,1>>>(d_map.begin(), d_map.end(), d_source.begin(), d_output.begin());
  
  ASSERT_EQUAL(h_output, d_output);
}
DECLARE_VARIABLE_UNITTEST(TestGatherDeviceSeq);


template<typename Iterator1, typename Iterator2, typename Iterator3, typename Iterator4, typename Predicate>
__global__
void gather_if_kernel(Iterator1 map_first, Iterator1 map_last, Iterator2 stencil_first, Iterator3 elements_first, Iterator4 result, Predicate pred)
{
  thrust::gather_if(thrust::seq, map_first, map_last, stencil_first, elements_first, result, pred);
}


template<typename T>
struct is_even_gather_if
{
  __host__ __device__
  bool operator()(const T i) const
  { 
    return (i % 2) == 0;
  }
};


template<typename T>
void TestGatherIfDeviceSeq(const size_t n)
{
  const size_t source_size = std::min((size_t) 10, 2 * n);
  
  // source vectors to gather from
  thrust::host_vector<T>   h_source = unittest::random_samples<T>(source_size);
  thrust::device_vector<T> d_source = h_source;
  
  // gather indices
  thrust::host_vector<unsigned int> h_map = unittest::random_integers<unsigned int>(n);
  
  for(size_t i = 0; i < n; i++)
      h_map[i] = h_map[i] % source_size;
  
  thrust::device_vector<unsigned int> d_map = h_map;
  
  // gather stencil
  thrust::host_vector<unsigned int> h_stencil = unittest::random_integers<unsigned int>(n);
  
  for(size_t i = 0; i < n; i++)
    h_stencil[i] = h_stencil[i] % 2;
  
  thrust::device_vector<unsigned int> d_stencil = h_stencil;
  
  // gather destination
  thrust::host_vector<T>   h_output(n);
  thrust::device_vector<T> d_output(n);
  
  thrust::gather_if(h_map.begin(), h_map.end(), h_stencil.begin(), h_source.begin(), h_output.begin(), is_even_gather_if<unsigned int>());
  gather_if_kernel<<<1,1>>>(d_map.begin(), d_map.end(), d_stencil.begin(), d_source.begin(), d_output.begin(), is_even_gather_if<unsigned int>());
  
  ASSERT_EQUAL(h_output, d_output);
}
DECLARE_VARIABLE_UNITTEST(TestGatherIfDeviceSeq);

