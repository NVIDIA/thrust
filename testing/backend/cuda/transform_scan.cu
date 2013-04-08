#include <unittest/unittest.h>
#include <thrust/transform_scan.h>
#include <thrust/execution_policy.h>


template<typename Iterator1, typename Iterator2, typename Function1, typename Function2, typename Iterator3>
__global__
void transform_inclusive_scan_kernel(Iterator1 first, Iterator1 last, Iterator2 result1, Function1 f1, Function2 f2, Iterator3 result2)
{
  *result2 = thrust::transform_inclusive_scan(thrust::seq, first, last, result1, f1, f2);
}


template<typename Iterator1, typename Iterator2, typename Function1, typename T, typename Function2, typename Iterator3>
__global__
void transform_exclusive_scan_kernel(Iterator1 first, Iterator1 last, Iterator2 result, Function1 f1, T init, Function2 f2, Iterator3 result2)
{
  *result2 = thrust::transform_exclusive_scan(thrust::seq, first, last, result, f1, init, f2);
}


void TestTransformScanDeviceSeq()
{
  typedef thrust::device_vector<int> Vector;
  typedef typename Vector::value_type T;
  
  typename Vector::iterator iter;
  
  Vector input(5);
  Vector ref(5);
  Vector output(5);
  
  input[0] = 1; input[1] = 3; input[2] = -2; input[3] = 4; input[4] = -5;
  
  Vector input_copy(input);

  thrust::device_vector<typename Vector::iterator> iter_vec(1);
  
  // inclusive scan
  transform_inclusive_scan_kernel<<<1,1>>>(input.begin(), input.end(), output.begin(), thrust::negate<T>(), thrust::plus<T>(), iter_vec.begin());
  iter = iter_vec[0];
  ref[0] = -1; ref[1] = -4; ref[2] = -2; ref[3] = -6; ref[4] = -1;
  ASSERT_EQUAL(iter - output.begin(), input.size());
  ASSERT_EQUAL(input,  input_copy);
  ASSERT_EQUAL(ref, output);
  
  // exclusive scan with 0 init
  transform_exclusive_scan_kernel<<<1,1>>>(input.begin(), input.end(), output.begin(), thrust::negate<T>(), 0, thrust::plus<T>(), iter_vec.begin());
  ref[0] = 0; ref[1] = -1; ref[2] = -4; ref[3] = -2; ref[4] = -6;
  ASSERT_EQUAL(iter - output.begin(), input.size());
  ASSERT_EQUAL(input,  input_copy);
  ASSERT_EQUAL(ref, output);
  
  // exclusive scan with nonzero init
  transform_exclusive_scan_kernel<<<1,1>>>(input.begin(), input.end(), output.begin(), thrust::negate<T>(), 3, thrust::plus<T>(), iter_vec.begin());
  iter = iter_vec[0];
  ref[0] = 3; ref[1] = 2; ref[2] = -1; ref[3] = 1; ref[4] = -3;
  ASSERT_EQUAL(iter - output.begin(), input.size());
  ASSERT_EQUAL(input,  input_copy);
  ASSERT_EQUAL(ref, output);
  
  // inplace inclusive scan
  input = input_copy;
  transform_inclusive_scan_kernel<<<1,1>>>(input.begin(), input.end(), input.begin(), thrust::negate<T>(), thrust::plus<T>(), iter_vec.begin());
  iter = iter_vec[0];
  ref[0] = -1; ref[1] = -4; ref[2] = -2; ref[3] = -6; ref[4] = -1;
  ASSERT_EQUAL(iter - input.begin(), input.size());
  ASSERT_EQUAL(ref, input);
  
  // inplace exclusive scan with init
  input = input_copy;
  transform_exclusive_scan_kernel<<<1,1>>>(input.begin(), input.end(), input.begin(), thrust::negate<T>(), 3, thrust::plus<T>(), iter_vec.begin());
  iter = iter_vec[0];
  ref[0] = 3; ref[1] = 2; ref[2] = -1; ref[3] = 1; ref[4] = -3;
  ASSERT_EQUAL(iter - input.begin(), input.size());
  ASSERT_EQUAL(ref, input);
}
DECLARE_UNITTEST(TestTransformScanDeviceSeq);

