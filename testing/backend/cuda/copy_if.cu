#include <unittest/unittest.h>
#include <thrust/adjacent_difference.h>
#include <thrust/execution_policy.h>


template<typename T>
struct is_even
{
  __host__ __device__
  bool operator()(T x) { return (static_cast<unsigned int>(x) & 1) == 0; }
};


template<typename T>
struct mod_3
{
  __host__ __device__
  unsigned int operator()(T x) { return static_cast<unsigned int>(x) % 3; }
};


template<typename ExecutionPolicy, typename Iterator1, typename Iterator2, typename Predicate, typename Iterator3>
__global__ void copy_if_kernel(ExecutionPolicy exec, Iterator1 first, Iterator1 last, Iterator2 result1, Predicate pred, Iterator3 result2)
{
  *result2 = thrust::copy_if(exec, first, last, result1, pred);
}


template<typename T, typename ExecutionPolicy>
void TestCopyIfDevice(ExecutionPolicy exec, const size_t n)
{
  thrust::host_vector<T>   h_data = unittest::random_integers<T>(n);
  thrust::device_vector<T> d_data = h_data;
  
  typename thrust::host_vector<T>::iterator   h_new_end;
  typename thrust::device_vector<T>::iterator d_new_end;

  thrust::device_vector<
    typename thrust::device_vector<T>::iterator
  > d_new_end_vec(1);
  
  // test with Predicate that returns a bool
  {
    thrust::host_vector<T>   h_result(n);
    thrust::device_vector<T> d_result(n);
    
    h_new_end = thrust::copy_if(h_data.begin(), h_data.end(), h_result.begin(), is_even<T>());
    copy_if_kernel<<<1,1>>>(exec, d_data.begin(), d_data.end(), d_result.begin(), is_even<T>(), d_new_end_vec.begin());
    d_new_end = d_new_end_vec[0];
    
    h_result.resize(h_new_end - h_result.begin());
    d_result.resize(d_new_end - d_result.begin());
    
    ASSERT_EQUAL(h_result, d_result);
  }
  
  // test with Predicate that returns a non-bool
  {
    thrust::host_vector<T>   h_result(n);
    thrust::device_vector<T> d_result(n);
    
    h_new_end = thrust::copy_if(h_data.begin(), h_data.end(), h_result.begin(), mod_3<T>());
    copy_if_kernel<<<1,1>>>(exec, d_data.begin(), d_data.end(), d_result.begin(), mod_3<T>(), d_new_end_vec.begin());
    d_new_end = d_new_end_vec[0];
    
    h_result.resize(h_new_end - h_result.begin());
    d_result.resize(d_new_end - d_result.begin());
    
    ASSERT_EQUAL(h_result, d_result);
  }
}


template<typename T>
void TestCopyIfDeviceSeq(const size_t n)
{
  TestCopyIfDevice<T>(thrust::seq, n);
}
DECLARE_VARIABLE_UNITTEST(TestCopyIfDeviceSeq);


template<typename T>
void TestCopyIfDeviceDevice(const size_t n)
{
  TestCopyIfDevice<T>(thrust::device, n);
}
DECLARE_VARIABLE_UNITTEST(TestCopyIfDeviceDevice);


template<typename ExecutionPolicy, typename Iterator1, typename Iterator2, typename Iterator3, typename Predicate, typename Iterator4>
__global__ void copy_if_kernel(ExecutionPolicy exec, Iterator1 first, Iterator1 last, Iterator2 stencil_first, Iterator3 result1, Predicate pred, Iterator4 result2)
{
  *result2 = thrust::copy_if(exec, first, last, stencil_first, result1, pred);
}


template<typename T, typename ExecutionPolicy>
void TestCopyIfStencilDevice(ExecutionPolicy exec, const size_t n)
{
  thrust::host_vector<T>   h_data(n); thrust::sequence(h_data.begin(), h_data.end());
  thrust::device_vector<T> d_data(n); thrust::sequence(d_data.begin(), d_data.end()); 
  
  thrust::host_vector<T>   h_stencil = unittest::random_integers<T>(n);
  thrust::device_vector<T> d_stencil = unittest::random_integers<T>(n);
  
  thrust::host_vector<T>   h_result(n);
  thrust::device_vector<T> d_result(n);
  
  typename thrust::host_vector<T>::iterator   h_new_end;
  typename thrust::device_vector<T>::iterator d_new_end;

  thrust::device_vector<
    typename thrust::device_vector<T>::iterator
  > d_new_end_vec(1);
  
  // test with Predicate that returns a bool
  {
    thrust::host_vector<T>   h_result(n);
    thrust::device_vector<T> d_result(n);
    
    h_new_end = thrust::copy_if(h_data.begin(), h_data.end(), h_result.begin(), is_even<T>());
    copy_if_kernel<<<1,1>>>(exec, d_data.begin(), d_data.end(), d_result.begin(), is_even<T>(), d_new_end_vec.begin());
    d_new_end = d_new_end_vec[0];
    
    h_result.resize(h_new_end - h_result.begin());
    d_result.resize(d_new_end - d_result.begin());
    
    ASSERT_EQUAL(h_result, d_result);
  }
  
  // test with Predicate that returns a non-bool
  {
    thrust::host_vector<T>   h_result(n);
    thrust::device_vector<T> d_result(n);
    
    h_new_end = thrust::copy_if(h_data.begin(), h_data.end(), h_result.begin(), mod_3<T>());
    copy_if_kernel<<<1,1>>>(exec, d_data.begin(), d_data.end(), d_result.begin(), mod_3<T>(), d_new_end_vec.begin());
    d_new_end = d_new_end_vec[0];
    
    h_result.resize(h_new_end - h_result.begin());
    d_result.resize(d_new_end - d_result.begin());
    
    ASSERT_EQUAL(h_result, d_result);
  }
}


template<typename T>
void TestCopyIfStencilDeviceSeq(const size_t n)
{
  TestCopyIfStencilDevice<T>(thrust::seq, n);
}
DECLARE_VARIABLE_UNITTEST(TestCopyIfStencilDeviceSeq);


template<typename T>
void TestCopyIfStencilDeviceDevice(const size_t n)
{
  TestCopyIfStencilDevice<T>(thrust::device, n);
}
DECLARE_VARIABLE_UNITTEST(TestCopyIfStencilDeviceDevice);


