#include <unittest/unittest.h>
#include <thrust/uninitialized_fill.h>
#include <thrust/execution_policy.h>


template<typename Iterator, typename T>
__global__
void uninitialized_fill_kernel(Iterator first, Iterator last, T val)
{
  thrust::uninitialized_fill(thrust::seq, first, last, val);
}


void TestUninitializedFillDeviceSeq()
{
  typedef thrust::device_vector<int> Vector;
  typedef Vector::value_type T;
  
  Vector v(5);
  v[0] = 0; v[1] = 1; v[2] = 2; v[3] = 3; v[4] = 4;
  
  T exemplar(7);
  
  uninitialized_fill_kernel<<<1,1>>>(v.begin() + 1, v.begin() + 4, exemplar);
  
  ASSERT_EQUAL(v[0], 0);
  ASSERT_EQUAL(v[1], exemplar);
  ASSERT_EQUAL(v[2], exemplar);
  ASSERT_EQUAL(v[3], exemplar);
  ASSERT_EQUAL(v[4], 4);
  
  exemplar = 8;
  
  uninitialized_fill_kernel<<<1,1>>>(v.begin() + 0, v.begin() + 3, exemplar);
  
  ASSERT_EQUAL(v[0], exemplar);
  ASSERT_EQUAL(v[1], exemplar);
  ASSERT_EQUAL(v[2], exemplar);
  ASSERT_EQUAL(v[3], 7);
  ASSERT_EQUAL(v[4], 4);
  
  exemplar = 9;
  
  uninitialized_fill_kernel<<<1,1>>>(v.begin() + 2, v.end(), exemplar);
  
  ASSERT_EQUAL(v[0], 8);
  ASSERT_EQUAL(v[1], 8);
  ASSERT_EQUAL(v[2], exemplar);
  ASSERT_EQUAL(v[3], exemplar);
  ASSERT_EQUAL(v[4], 9);
  
  exemplar = 1;
  
  uninitialized_fill_kernel<<<1,1>>>(v.begin(), v.end(), exemplar);
  
  ASSERT_EQUAL(v[0], exemplar);
  ASSERT_EQUAL(v[1], exemplar);
  ASSERT_EQUAL(v[2], exemplar);
  ASSERT_EQUAL(v[3], exemplar);
  ASSERT_EQUAL(v[4], exemplar);
}
DECLARE_UNITTEST(TestUninitializedFillDeviceSeq);


template<typename Iterator1, typename Size, typename T, typename Iterator2>
__global__
void uninitialized_fill_n_kernel(Iterator1 first, Size n, T val, Iterator2 result)
{
  *result = thrust::uninitialized_fill_n(thrust::seq, first, n, val);
}


void TestUninitializedFillNDeviceSeq()
{
  typedef thrust::device_vector<int> Vector;
  typedef Vector::value_type T;
  
  Vector v(5);
  v[0] = 0; v[1] = 1; v[2] = 2; v[3] = 3; v[4] = 4;
  
  T exemplar(7);

  thrust::device_vector<Vector::iterator> iter_vec(1);
  
  uninitialized_fill_n_kernel<<<1,1>>>(v.begin() + 1, 3, exemplar, iter_vec.begin());
  Vector::iterator iter = iter_vec[0];
  
  ASSERT_EQUAL(v[0], 0);
  ASSERT_EQUAL(v[1], exemplar);
  ASSERT_EQUAL(v[2], exemplar);
  ASSERT_EQUAL(v[3], exemplar);
  ASSERT_EQUAL(v[4], 4);
  ASSERT_EQUAL_QUIET(v.begin() + 4, iter);
  
  exemplar = 8;
  
  uninitialized_fill_n_kernel<<<1,1>>>(v.begin() + 0, 3, exemplar, iter_vec.begin());
  iter = iter_vec[0];
  
  ASSERT_EQUAL(v[0], exemplar);
  ASSERT_EQUAL(v[1], exemplar);
  ASSERT_EQUAL(v[2], exemplar);
  ASSERT_EQUAL(v[3], 7);
  ASSERT_EQUAL(v[4], 4);
  ASSERT_EQUAL_QUIET(v.begin() + 3, iter);
  
  exemplar = 9;
  
  uninitialized_fill_n_kernel<<<1,1>>>(v.begin() + 2, 3, exemplar, iter_vec.begin());
  iter = iter_vec[0];
  
  ASSERT_EQUAL(v[0], 8);
  ASSERT_EQUAL(v[1], 8);
  ASSERT_EQUAL(v[2], exemplar);
  ASSERT_EQUAL(v[3], exemplar);
  ASSERT_EQUAL(v[4], 9);
  ASSERT_EQUAL_QUIET(v.end(), iter);
  
  exemplar = 1;
  
  uninitialized_fill_n_kernel<<<1,1>>>(v.begin(), v.size(), exemplar, iter_vec.begin());
  iter = iter_vec[0];
  
  ASSERT_EQUAL(v[0], exemplar);
  ASSERT_EQUAL(v[1], exemplar);
  ASSERT_EQUAL(v[2], exemplar);
  ASSERT_EQUAL(v[3], exemplar);
  ASSERT_EQUAL(v[4], exemplar);
  ASSERT_EQUAL_QUIET(v.end(), iter);
}
DECLARE_UNITTEST(TestUninitializedFillNDeviceSeq);

