#include <unittest/unittest.h>
#include <thrust/remove.h>
#include <thrust/execution_policy.h>


template<typename ExecutionPolicy, typename Iterator, typename T, typename Iterator2>
__global__
void remove_kernel(ExecutionPolicy exec, Iterator first, Iterator last, T val, Iterator2 result)
{
  *result = thrust::remove(exec, first, last, val);
}


template<typename ExecutionPolicy, typename Iterator, typename Predicate, typename Iterator2>
__global__
void remove_if_kernel(ExecutionPolicy exec, Iterator first, Iterator last, Predicate pred, Iterator2 result)
{
  *result = thrust::remove_if(exec, first, last, pred);
}


template<typename ExecutionPolicy, typename Iterator1, typename Iterator2, typename Predicate, typename Iterator3>
__global__
void remove_if_kernel(ExecutionPolicy exec, Iterator1 first, Iterator1 last, Iterator2 stencil_first, Predicate pred, Iterator3 result)
{
  *result = thrust::remove_if(exec, first, last, stencil_first, pred);
}


template<typename ExecutionPolicy, typename Iterator1, typename Iterator2, typename T, typename Iterator3>
__global__
void remove_copy_kernel(ExecutionPolicy exec, Iterator1 first, Iterator1 last, Iterator2 result1, T val, Iterator3 result2)
{
  *result2 = thrust::remove_copy(exec, first, last, result1, val);
}


template<typename ExecutionPolicy, typename Iterator1, typename Iterator2, typename Predicate, typename Iterator3>
__global__
void remove_copy_if_kernel(ExecutionPolicy exec, Iterator1 first, Iterator1 last, Iterator2 result, Predicate pred, Iterator3 result_end)
{
  *result_end = thrust::remove_copy_if(exec, first, last, result, pred);
}


template<typename ExecutionPolicy, typename Iterator1, typename Iterator2, typename Iterator3, typename Predicate, typename Iterator4>
__global__
void remove_copy_if_kernel(ExecutionPolicy exec, Iterator1 first, Iterator1 last, Iterator2 stencil_first, Iterator3 result, Predicate pred, Iterator4 result_end)
{
  *result_end = thrust::remove_copy_if(exec, first, last, stencil_first, result, pred);
}


template<typename T>
struct is_even
  : thrust::unary_function<T,bool>
{
  __host__ __device__
  bool operator()(T x) { return (static_cast<unsigned int>(x) & 1) == 0; }
};


template<typename T>
struct is_true
  : thrust::unary_function<T,bool>
{
  __host__ __device__
  bool operator()(T x) { return x ? true : false; }
};


template<typename T, typename ExecutionPolicy>
void TestRemoveDevice(ExecutionPolicy exec, const size_t n)
{
  thrust::host_vector<T>   h_data = unittest::random_samples<T>(n);
  thrust::device_vector<T> d_data = h_data;

  typedef typename thrust::device_vector<T>::iterator iterator;
  thrust::device_vector<iterator> d_result(1);
  
  size_t h_size = thrust::remove(h_data.begin(), h_data.end(), T(0)) - h_data.begin();
  remove_kernel<<<1,1>>>(exec, d_data.begin(), d_data.end(), T(0), d_result.begin());
  size_t d_size = (iterator)d_result[0] - d_data.begin();
  
  ASSERT_EQUAL(h_size, d_size);
  
  h_data.resize(h_size);
  d_data.resize(d_size);
  
  ASSERT_EQUAL(h_data, d_data);
}


template<typename T>
void TestRemoveDeviceSeq(const size_t n)
{
  TestRemoveDevice<T>(thrust::seq, n);
}
DECLARE_VARIABLE_UNITTEST(TestRemoveDeviceSeq);


template<typename T>
void TestRemoveDeviceDevice(const size_t n)
{
  TestRemoveDevice<T>(thrust::device, n);
}
DECLARE_VARIABLE_UNITTEST(TestRemoveDeviceDevice);


template<typename T, typename ExecutionPolicy>
void TestRemoveIfDevice(ExecutionPolicy exec, const size_t n)
{
  thrust::host_vector<T>   h_data = unittest::random_samples<T>(n);
  thrust::device_vector<T> d_data = h_data;

  typedef typename thrust::device_vector<T>::iterator iterator;
  thrust::device_vector<iterator> d_result(1);
  
  size_t h_size = thrust::remove_if(h_data.begin(), h_data.end(), is_true<T>()) - h_data.begin();
  remove_if_kernel<<<1,1>>>(exec, d_data.begin(), d_data.end(), is_true<T>(), d_result.begin());
  size_t d_size = (iterator)d_result[0] - d_data.begin();
  
  ASSERT_EQUAL(h_size, d_size);
  
  h_data.resize(h_size);
  d_data.resize(d_size);
  
  ASSERT_EQUAL(h_data, d_data);
}


template<typename T>
void TestRemoveIfDeviceSeq(const size_t n)
{
  TestRemoveIfDevice<T>(thrust::seq, n);
}
DECLARE_VARIABLE_UNITTEST(TestRemoveIfDeviceSeq);


template<typename T>
void TestRemoveIfDeviceDevice(const size_t n)
{
  TestRemoveIfDevice<T>(thrust::device, n);
}
DECLARE_VARIABLE_UNITTEST(TestRemoveIfDeviceDevice);


template<typename T, typename ExecutionPolicy>
void TestRemoveIfStencilDevice(ExecutionPolicy exec, const size_t n)
{
  thrust::host_vector<T>   h_data = unittest::random_samples<T>(n);
  thrust::device_vector<T> d_data = h_data;

  typedef typename thrust::device_vector<T>::iterator iterator;
  thrust::device_vector<iterator> d_result(1);
  
  thrust::host_vector<bool>   h_stencil = unittest::random_integers<bool>(n);
  thrust::device_vector<bool> d_stencil = h_stencil;
  
  size_t h_size = thrust::remove_if(h_data.begin(), h_data.end(), h_stencil.begin(), is_true<T>()) - h_data.begin();

  remove_if_kernel<<<1,1>>>(exec, d_data.begin(), d_data.end(), d_stencil.begin(), is_true<T>(), d_result.begin());
  size_t d_size = (iterator)d_result[0] - d_data.begin();
  
  ASSERT_EQUAL(h_size, d_size);
  
  h_data.resize(h_size);
  d_data.resize(d_size);
  
  ASSERT_EQUAL(h_data, d_data);
}


template<typename T>
void TestRemoveIfStencilDeviceSeq(const size_t n)
{
  TestRemoveIfStencilDevice<T>(thrust::seq, n);
}
DECLARE_VARIABLE_UNITTEST(TestRemoveIfStencilDeviceSeq);


template<typename T>
void TestRemoveIfStencilDeviceDevice(const size_t n)
{
  TestRemoveIfStencilDevice<T>(thrust::device, n);
}
DECLARE_VARIABLE_UNITTEST(TestRemoveIfStencilDeviceDevice);


template<typename T, typename ExecutionPolicy>
void TestRemoveCopyDevice(ExecutionPolicy exec, const size_t n)
{
  thrust::host_vector<T>   h_data = unittest::random_samples<T>(n);
  thrust::device_vector<T> d_data = h_data;
  
  thrust::host_vector<T>   h_result(n);
  thrust::device_vector<T> d_result(n);

  typedef typename thrust::device_vector<T>::iterator iterator;
  thrust::device_vector<iterator> d_new_end(1);
  
  size_t h_size = thrust::remove_copy(h_data.begin(), h_data.end(), h_result.begin(), T(0)) - h_result.begin();

  remove_copy_kernel<<<1,1>>>(exec, d_data.begin(), d_data.end(), d_result.begin(), T(0), d_new_end.begin());
  size_t d_size = (iterator)d_new_end[0] - d_result.begin();
  
  ASSERT_EQUAL(h_size, d_size);
  
  h_result.resize(h_size);
  d_result.resize(d_size);
  
  ASSERT_EQUAL(h_result, d_result);
}


template<typename T>
void TestRemoveCopyDeviceSeq(const size_t n)
{
  TestRemoveCopyDevice<T>(thrust::seq, n);
}
DECLARE_VARIABLE_UNITTEST(TestRemoveCopyDeviceSeq);


template<typename T>
void TestRemoveCopyDeviceDevice(const size_t n)
{
  TestRemoveCopyDevice<T>(thrust::device, n);
}
DECLARE_VARIABLE_UNITTEST(TestRemoveCopyDeviceDevice);


template<typename T, typename ExecutionPolicy>
void TestRemoveCopyIfDevice(ExecutionPolicy exec, const size_t n)
{
  thrust::host_vector<T>   h_data = unittest::random_samples<T>(n);
  thrust::device_vector<T> d_data = h_data;
  
  thrust::host_vector<T>   h_result(n);
  thrust::device_vector<T> d_result(n);

  typedef typename thrust::device_vector<T>::iterator iterator;
  thrust::device_vector<iterator> d_new_end(1);
  
  size_t h_size = thrust::remove_copy_if(h_data.begin(), h_data.end(), h_result.begin(), is_true<T>()) - h_result.begin();

  remove_copy_if_kernel<<<1,1>>>(exec, d_data.begin(), d_data.end(), d_result.begin(), is_true<T>(), d_new_end.begin());
  size_t d_size = (iterator)d_new_end[0] - d_result.begin();
  
  ASSERT_EQUAL(h_size, d_size);
  
  h_result.resize(h_size);
  d_result.resize(d_size);
  
  ASSERT_EQUAL(h_result, d_result);
}


template<typename T>
void TestRemoveCopyIfDeviceSeq(const size_t n)
{
  TestRemoveCopyIfDevice<T>(thrust::seq, n);
}
DECLARE_VARIABLE_UNITTEST(TestRemoveCopyIfDeviceSeq);


template<typename T>
void TestRemoveCopyIfDeviceDevice(const size_t n)
{
  TestRemoveCopyIfDevice<T>(thrust::device, n);
}
DECLARE_VARIABLE_UNITTEST(TestRemoveCopyIfDeviceDevice);


template<typename T, typename ExecutionPolicy>
void TestRemoveCopyIfStencilDevice(ExecutionPolicy exec, const size_t n)
{
  thrust::host_vector<T>   h_data = unittest::random_samples<T>(n);
  thrust::device_vector<T> d_data = h_data;
  
  thrust::host_vector<T>   h_result(n);
  thrust::device_vector<T> d_result(n);

  typedef typename thrust::device_vector<T>::iterator iterator;
  thrust::device_vector<iterator> d_new_end(1);

  thrust::host_vector<bool>   h_stencil = unittest::random_integers<bool>(n);
  thrust::device_vector<bool> d_stencil = h_stencil;
  
  size_t h_size = thrust::remove_copy_if(h_data.begin(), h_data.end(), h_stencil.begin(), h_result.begin(), is_true<T>()) - h_result.begin();

  remove_copy_if_kernel<<<1,1>>>(exec, d_data.begin(), d_data.end(), d_stencil.begin(), d_result.begin(), is_true<T>(), d_new_end.begin());
  size_t d_size = (iterator)d_new_end[0] - d_result.begin();
  
  ASSERT_EQUAL(h_size, d_size);
  
  h_result.resize(h_size);
  d_result.resize(d_size);
  
  ASSERT_EQUAL(h_result, d_result);
}


template<typename T>
void TestRemoveCopyIfStencilDeviceSeq(const size_t n)
{
  TestRemoveCopyIfStencilDevice<T>(thrust::seq, n);
}
DECLARE_VARIABLE_UNITTEST(TestRemoveCopyIfStencilDeviceSeq);


template<typename T>
void TestRemoveCopyIfStencilDeviceDevice(const size_t n)
{
  TestRemoveCopyIfStencilDevice<T>(thrust::device, n);
}
DECLARE_VARIABLE_UNITTEST(TestRemoveCopyIfStencilDeviceDevice);

