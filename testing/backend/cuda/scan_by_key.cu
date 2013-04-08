#include <unittest/unittest.h>
#include <thrust/scan.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>


template<typename Iterator1, typename Iterator2, typename Iterator3>
__global__
void inclusive_scan_by_key_kernel(Iterator1 keys_first, Iterator1 keys_last, Iterator2 values_first, Iterator3 result)
{
  thrust::inclusive_scan_by_key(thrust::seq, keys_first, keys_last, values_first, result);
}


template<typename Iterator1, typename Iterator2, typename Iterator3>
__global__
void exclusive_scan_by_key_kernel(Iterator1 keys_first, Iterator1 keys_last, Iterator2 values_first, Iterator3 result)
{
  thrust::exclusive_scan_by_key(thrust::seq, keys_first, keys_last, values_first, result);
}


template<typename Iterator1, typename Iterator2, typename Iterator3, typename T>
__global__
void exclusive_scan_by_key_kernel(Iterator1 keys_first, Iterator1 keys_last, Iterator2 values_first, Iterator3 result, T init)
{
  thrust::exclusive_scan_by_key(thrust::seq, keys_first, keys_last, values_first, result, init);
}


template<typename T>
void TestScanByKeyDeviceSeq(const size_t n)
{
  thrust::host_vector<int> h_keys(n);
  for(size_t i = 0, k = 0; i < n; i++)
  {
    h_keys[i] = k;
    if(rand() % 10 == 0)
    {
      k++;
    }
  }
  thrust::device_vector<int> d_keys = h_keys;
  
  thrust::host_vector<T>   h_vals = unittest::random_integers<int>(n);
  for(size_t i = 0; i < n; i++)
  {
    h_vals[i] = i % 10;
  }
  thrust::device_vector<T> d_vals = h_vals;
  
  thrust::host_vector<T>   h_output(n);
  thrust::device_vector<T> d_output(n);
  
  thrust::inclusive_scan_by_key(h_keys.begin(), h_keys.end(), h_vals.begin(), h_output.begin());
  inclusive_scan_by_key_kernel<<<1,1>>>(d_keys.begin(), d_keys.end(), d_vals.begin(), d_output.begin());
  ASSERT_EQUAL(d_output, h_output);
  
  thrust::exclusive_scan_by_key(h_keys.begin(), h_keys.end(), h_vals.begin(), h_output.begin());
  exclusive_scan_by_key_kernel<<<1,1>>>(d_keys.begin(), d_keys.end(), d_vals.begin(), d_output.begin());
  ASSERT_EQUAL(d_output, h_output);
  
  thrust::exclusive_scan_by_key(h_keys.begin(), h_keys.end(), h_vals.begin(), h_output.begin(), (T) 11);
  exclusive_scan_by_key_kernel<<<1,1>>>(d_keys.begin(), d_keys.end(), d_vals.begin(), d_output.begin(), (T) 11);
  ASSERT_EQUAL(d_output, h_output);
  
  // in-place scans
  h_output = h_vals;
  d_output = d_vals;
  thrust::inclusive_scan_by_key(h_keys.begin(), h_keys.end(), h_output.begin(), h_output.begin());
  inclusive_scan_by_key_kernel<<<1,1>>>(d_keys.begin(), d_keys.end(), d_output.begin(), d_output.begin());
  ASSERT_EQUAL(d_output, h_output);
  
  h_output = h_vals;
  d_output = d_vals;
  thrust::exclusive_scan_by_key(h_keys.begin(), h_keys.end(), h_output.begin(), h_output.begin(), (T) 11);
  exclusive_scan_by_key_kernel<<<1,1>>>(d_keys.begin(), d_keys.end(), d_output.begin(), d_output.begin(), (T) 11);
  ASSERT_EQUAL(d_output, h_output);
}
DECLARE_VARIABLE_UNITTEST(TestScanByKeyDeviceSeq);

