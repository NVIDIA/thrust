#include <unittest/unittest.h>
#include <thrust/fill.h>
#include <thrust/execution_policy.h>


template<typename Iterator, typename T>
__global__
void fill_kernel(Iterator first, Iterator last, T value)
{
  thrust::fill(thrust::seq, first, last, value);
}


template<typename T>
void TestFillDeviceSeq(size_t n)
{
  thrust::host_vector<T>   h_data = unittest::random_integers<T>(n);
  thrust::device_vector<T> d_data = h_data;
  
  thrust::fill(h_data.begin() + std::min((size_t)1, n), h_data.begin() + std::min((size_t)3, n), (T) 0);
  fill_kernel<<<1,1>>>(d_data.begin() + std::min((size_t)1, n), d_data.begin() + std::min((size_t)3, n), (T) 0);
  
  ASSERT_EQUAL(h_data, d_data);
  
  thrust::fill(h_data.begin() + std::min((size_t)117, n), h_data.begin() + std::min((size_t)367, n), (T) 1);
  fill_kernel<<<1,1>>>(d_data.begin() + std::min((size_t)117, n), d_data.begin() + std::min((size_t)367, n), (T) 1);
  
  ASSERT_EQUAL(h_data, d_data);
  
  thrust::fill(h_data.begin() + std::min((size_t)8, n), h_data.begin() + std::min((size_t)259, n), (T) 2);
  fill_kernel<<<1,1>>>(d_data.begin() + std::min((size_t)8, n), d_data.begin() + std::min((size_t)259, n), (T) 2);
  
  ASSERT_EQUAL(h_data, d_data);
  
  thrust::fill(h_data.begin() + std::min((size_t)3, n), h_data.end(), (T) 3);
  fill_kernel<<<1,1>>>(d_data.begin() + std::min((size_t)3, n), d_data.end(), (T) 3);
  
  ASSERT_EQUAL(h_data, d_data);
  
  thrust::fill(h_data.begin(), h_data.end(), (T) 4);
  fill_kernel<<<1,1>>>(d_data.begin(), d_data.end(), (T) 4);
  
  ASSERT_EQUAL(h_data, d_data);
}
DECLARE_VARIABLE_UNITTEST(TestFillDeviceSeq);


template<typename Iterator, typename Size, typename T>
__global__
void fill_n_kernel(Iterator first, Size n, T value)
{
  thrust::fill_n(thrust::seq, first, n, value);
}


template<typename T>
void TestFillNDeviceSeq(size_t n)
{
  thrust::host_vector<T>   h_data = unittest::random_integers<T>(n);
  thrust::device_vector<T> d_data = h_data;
  
  size_t begin_offset = std::min<size_t>(1,n);
  thrust::fill_n(h_data.begin() + begin_offset, std::min((size_t)3, n) - begin_offset, (T) 0);
  fill_n_kernel<<<1,1>>>(d_data.begin() + begin_offset, std::min((size_t)3, n) - begin_offset, (T) 0);
  
  ASSERT_EQUAL(h_data, d_data);
  
  begin_offset = std::min<size_t>(117, n);
  thrust::fill_n(h_data.begin() + begin_offset, std::min((size_t)367, n) - begin_offset, (T) 1);
  fill_n_kernel<<<1,1>>>(d_data.begin() + begin_offset, std::min((size_t)367, n) - begin_offset, (T) 1);
  
  ASSERT_EQUAL(h_data, d_data);
  
  begin_offset = std::min<size_t>(8, n);
  thrust::fill_n(h_data.begin() + begin_offset, std::min((size_t)259, n) - begin_offset, (T) 2);
  fill_n_kernel<<<1,1>>>(d_data.begin() + begin_offset, std::min((size_t)259, n) - begin_offset, (T) 2);
  
  ASSERT_EQUAL(h_data, d_data);
  
  begin_offset = std::min<size_t>(3, n);
  thrust::fill_n(h_data.begin() + begin_offset, h_data.size() - begin_offset, (T) 3);
  fill_n_kernel<<<1,1>>>(d_data.begin() + begin_offset, d_data.size() - begin_offset, (T) 3);
  
  ASSERT_EQUAL(h_data, d_data);
  
  thrust::fill_n(h_data.begin(), h_data.size(), (T) 4);
  fill_n_kernel<<<1,1>>>(d_data.begin(), d_data.size(), (T) 4);
  
  ASSERT_EQUAL(h_data, d_data);
}
DECLARE_VARIABLE_UNITTEST(TestFillNDeviceSeq);

