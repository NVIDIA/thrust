#include <unittest/unittest.h>
#include <thrust/replace.h>
#include <thrust/execution_policy.h>


template <typename T>
struct less_than_five
{
  __host__ __device__ bool operator()(const T &val) const {return val < 5;}
};


template<typename ExecutionPolicy, typename Iterator, typename T1, typename T2>
__global__
void replace_kernel(ExecutionPolicy exec, Iterator first, Iterator last, T1 old_value, T2 new_value)
{
  thrust::replace(exec, first, last, old_value, new_value);
}


template<typename T, typename ExecutionPolicy>
void TestReplaceDevice(ExecutionPolicy exec, const size_t n)
{
  thrust::host_vector<T>   h_data = unittest::random_samples<T>(n);
  thrust::device_vector<T> d_data = h_data;
  
  T old_value = 0;
  T new_value = 1;
  
  thrust::replace(h_data.begin(), h_data.end(), old_value, new_value);
  replace_kernel<<<1,1>>>(exec, d_data.begin(), d_data.end(), old_value, new_value);
  
  ASSERT_ALMOST_EQUAL(h_data, d_data);
}


template<typename T>
void TestReplaceDeviceSeq(const size_t n)
{
  TestReplaceDevice<T>(thrust::seq, n);
}
DECLARE_VARIABLE_UNITTEST(TestReplaceDeviceSeq);

template<typename T>
void TestReplaceDeviceDevice(const size_t n)
{
  TestReplaceDevice<T>(thrust::device, n);
}
DECLARE_VARIABLE_UNITTEST(TestReplaceDeviceDevice);


template<typename ExecutionPolicy, typename Iterator1, typename Iterator2, typename T1, typename T2>
__global__
void replace_copy_kernel(ExecutionPolicy exec, Iterator1 first, Iterator1 last, Iterator2 result, T1 old_value, T2 new_value)
{
  thrust::replace_copy(exec, first, last, result, old_value, new_value);
}


template<typename T, typename ExecutionPolicy>
void TestReplaceCopyDevice(ExecutionPolicy exec, const size_t n)
{
  thrust::host_vector<T>   h_data = unittest::random_samples<T>(n);
  thrust::device_vector<T> d_data = h_data;
  
  T old_value = 0;
  T new_value = 1;
  
  thrust::host_vector<T>   h_dest(n);
  thrust::device_vector<T> d_dest(n);
  
  thrust::replace_copy(h_data.begin(), h_data.end(), h_dest.begin(), old_value, new_value);
  replace_copy_kernel<<<1,1>>>(exec, d_data.begin(), d_data.end(), d_dest.begin(), old_value, new_value);
  
  ASSERT_ALMOST_EQUAL(h_data, d_data);
  ASSERT_ALMOST_EQUAL(h_dest, d_dest);
}

template<typename T>
void TestReplaceCopyDeviceSeq(const size_t n)
{
  TestReplaceCopyDevice<T>(thrust::seq, n);
}
DECLARE_VARIABLE_UNITTEST(TestReplaceCopyDeviceSeq);

template<typename T>
void TestReplaceCopyDeviceDevice(const size_t n)
{
  TestReplaceCopyDevice<T>(thrust::device, n);
}
DECLARE_VARIABLE_UNITTEST(TestReplaceCopyDeviceDevice);


template<typename ExecutionPolicy, typename Iterator, typename Predicate, typename T>
__global__
void replace_if_kernel(ExecutionPolicy exec, Iterator first, Iterator last, Predicate pred, T new_value)
{
  thrust::replace_if(exec, first, last, pred, new_value);
}


template<typename T, typename ExecutionPolicy>
void TestReplaceIfDevice(ExecutionPolicy exec, const size_t n)
{
  thrust::host_vector<T>   h_data = unittest::random_samples<T>(n);
  thrust::device_vector<T> d_data = h_data;
  
  thrust::replace_if(h_data.begin(), h_data.end(), less_than_five<T>(), (T) 0);
  replace_if_kernel<<<1,1>>>(exec, d_data.begin(), d_data.end(), less_than_five<T>(), (T) 0);
  
  ASSERT_ALMOST_EQUAL(h_data, d_data);
}

template<typename T>
void TestReplaceIfDeviceSeq(const size_t n)
{
  TestReplaceIfDevice<T>(thrust::seq, n);
}
DECLARE_VARIABLE_UNITTEST(TestReplaceIfDeviceSeq);

template<typename T>
void TestReplaceIfDeviceDevice(const size_t n)
{
  TestReplaceIfDevice<T>(thrust::device, n);
}
DECLARE_VARIABLE_UNITTEST(TestReplaceIfDeviceDevice);


template<typename ExecutionPolicy, typename Iterator1, typename Iterator2, typename Predicate, typename T>
__global__
void replace_if_kernel(ExecutionPolicy exec, Iterator1 first, Iterator1 last, Iterator2 stencil_first, Predicate pred, T new_value)
{
  thrust::replace_if(exec, first, last, stencil_first, pred, new_value);
}


template<typename T, typename ExecutionPolicy>
void TestReplaceIfStencilDevice(ExecutionPolicy exec, const size_t n)
{
  thrust::host_vector<T>   h_data = unittest::random_samples<T>(n);
  thrust::device_vector<T> d_data = h_data;
  
  thrust::host_vector<T>   h_stencil = unittest::random_samples<T>(n);
  thrust::device_vector<T> d_stencil = h_stencil;
  
  thrust::replace_if(h_data.begin(), h_data.end(), h_stencil.begin(), less_than_five<T>(), (T) 0);
  replace_if_kernel<<<1,1>>>(exec, d_data.begin(), d_data.end(), d_stencil.begin(), less_than_five<T>(), (T) 0);
  
  ASSERT_ALMOST_EQUAL(h_data, d_data);
}

template<typename T>
void TestReplaceIfStencilDeviceSeq(const size_t n)
{
  TestReplaceIfStencilDevice<T>(thrust::seq, n);
}
DECLARE_VARIABLE_UNITTEST(TestReplaceIfStencilDeviceSeq);

template<typename T>
void TestReplaceIfStencilDeviceDevice(const size_t n)
{
  TestReplaceIfStencilDevice<T>(thrust::device, n);
}
DECLARE_VARIABLE_UNITTEST(TestReplaceIfStencilDeviceDevice);


template<typename ExecutionPolicy, typename Iterator1, typename Iterator2, typename Predicate, typename T>
__global__
void replace_copy_if_kernel(ExecutionPolicy exec, Iterator1 first, Iterator1 last, Iterator2 result, Predicate pred, T new_value)
{
  thrust::replace_copy_if(exec, first, last, result, pred, new_value);
}


template<typename T, typename ExecutionPolicy>
void TestReplaceCopyIfDevice(ExecutionPolicy exec, const size_t n)
{
  thrust::host_vector<T>   h_data = unittest::random_samples<T>(n);
  thrust::device_vector<T> d_data = h_data;
  
  thrust::host_vector<T>   h_dest(n);
  thrust::device_vector<T> d_dest(n);
  
  thrust::replace_copy_if(h_data.begin(), h_data.end(), h_dest.begin(), less_than_five<T>(), 0);
  replace_copy_if_kernel<<<1,1>>>(exec, d_data.begin(), d_data.end(), d_dest.begin(), less_than_five<T>(), 0);
  
  ASSERT_ALMOST_EQUAL(h_data, d_data);
  ASSERT_ALMOST_EQUAL(h_dest, d_dest);
}

template<typename T>
void TestReplaceCopyIfDeviceSeq(const size_t n)
{
  TestReplaceCopyIfDevice<T>(thrust::seq, n);
}
DECLARE_VARIABLE_UNITTEST(TestReplaceCopyIfDeviceSeq);

template<typename T>
void TestReplaceCopyIfDeviceDevice(const size_t n)
{
  TestReplaceCopyIfDevice<T>(thrust::device, n);
}
DECLARE_VARIABLE_UNITTEST(TestReplaceCopyIfDeviceDevice);


template<typename ExecutionPolicy, typename Iterator1, typename Iterator2, typename Iterator3, typename Predicate, typename T>
__global__
void replace_copy_if_kernel(ExecutionPolicy exec, Iterator1 first, Iterator1 last, Iterator2 stencil_first, Iterator3 result, Predicate pred, T new_value)
{
  thrust::replace_copy_if(exec, first, last, stencil_first, result, pred, new_value);
}


template<typename T, typename ExecutionPolicy>
void TestReplaceCopyIfStencilDevice(ExecutionPolicy exec, const size_t n)
{
  thrust::host_vector<T>   h_data = unittest::random_samples<T>(n);
  thrust::device_vector<T> d_data = h_data;
  
  thrust::host_vector<T>   h_stencil = unittest::random_samples<T>(n);
  thrust::device_vector<T> d_stencil = h_stencil;
  
  thrust::host_vector<T>   h_dest(n);
  thrust::device_vector<T> d_dest(n);
  
  thrust::replace_copy_if(h_data.begin(), h_data.end(), h_stencil.begin(), h_dest.begin(), less_than_five<T>(), 0);
  replace_copy_if_kernel<<<1,1>>>(exec, d_data.begin(), d_data.end(), d_stencil.begin(), d_dest.begin(), less_than_five<T>(), 0);
  
  ASSERT_ALMOST_EQUAL(h_data, d_data);
  ASSERT_ALMOST_EQUAL(h_dest, d_dest);
}

template<typename T>
void TestReplaceCopyIfStencilDeviceSeq(const size_t n)
{
  TestReplaceCopyIfStencilDevice<T>(thrust::seq, n);
}
DECLARE_VARIABLE_UNITTEST(TestReplaceCopyIfStencilDeviceSeq);

template<typename T>
void TestReplaceCopyIfStencilDeviceDevice(const size_t n)
{
  TestReplaceCopyIfStencilDevice<T>(thrust::device, n);
}
DECLARE_VARIABLE_UNITTEST(TestReplaceCopyIfStencilDeviceDevice);

