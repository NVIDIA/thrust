#include <unittest/unittest.h>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>


template<typename Iterator1, typename Iterator2, typename Function, typename Iterator3>
__global__
void transform_kernel(Iterator1 first, Iterator1 last, Iterator2 result1, Function f, Iterator3 result2)
{
  *result2 = thrust::transform(thrust::seq, first, last, result1, f);
}


void TestTransformUnaryDeviceSeq()
{
  typedef thrust::device_vector<int> Vector;
  typedef typename Vector::value_type T;
  
  typename Vector::iterator iter;
  
  Vector input(3);
  Vector output(3);
  Vector result(3);
  input[0]  =  1; input[1]  = -2; input[2]  =  3;
  result[0] = -1; result[1] =  2; result[2] = -3;

  thrust::device_vector<typename Vector::iterator> iter_vec(1);
  
  transform_kernel<<<1,1>>>(input.begin(), input.end(), output.begin(), thrust::negate<T>(), iter_vec.begin());
  iter = iter_vec[0];
  
  ASSERT_EQUAL(iter - output.begin(), input.size());
  ASSERT_EQUAL(output, result);
}
DECLARE_UNITTEST(TestTransformUnaryDeviceSeq);


template<typename Iterator1, typename Iterator2, typename Function, typename Predicate, typename Iterator3>
__global__
void transform_if_kernel(Iterator1 first, Iterator1 last, Iterator2 result1, Function f, Predicate pred, Iterator3 result2)
{
  *result2 = thrust::transform_if(thrust::seq, first, last, result1, f, pred);
}


void TestTransformIfUnaryNoStencilDeviceSeq()
{
  typedef thrust::device_vector<int> Vector;
  typedef typename Vector::value_type T;
  
  typename Vector::iterator iter;
  
  Vector input(3);
  Vector output(3);
  Vector result(3);
  
  input[0]   =  0; input[1]   = -2; input[2]   =  0;
  output[0]  = -1; output[1]  = -2; output[2]  = -3; 
  result[0]  = -1; result[1]  =  2; result[2]  = -3;

  thrust::device_vector<typename Vector::iterator> iter_vec(1);
  
  transform_if_kernel<<<1,1>>>(input.begin(), input.end(),
                               output.begin(),
                               thrust::negate<T>(),
                               thrust::identity<T>(),
                               iter_vec.begin());
  iter = iter_vec[0];
  
  ASSERT_EQUAL(iter - output.begin(), input.size());
  ASSERT_EQUAL(output, result);
}
DECLARE_UNITTEST(TestTransformIfUnaryNoStencilDeviceSeq);


template<typename Iterator1, typename Iterator2, typename Iterator3, typename Function, typename Predicate, typename Iterator4>
__global__
void transform_if_kernel(Iterator1 first, Iterator1 last, Iterator2 stencil_first, Iterator3 result1, Function f, Predicate pred, Iterator4 result2)
{
  *result2 = thrust::transform_if(thrust::seq, first, last, stencil_first, result1, f, pred);
}


void TestTransformIfUnaryDeviceSeq()
{
  typedef thrust::device_vector<int> Vector;
  typedef typename Vector::value_type T;
  
  typename Vector::iterator iter;
  
  Vector input(3);
  Vector stencil(3);
  Vector output(3);
  Vector result(3);
  
  input[0]   =  1; input[1]   = -2; input[2]   =  3;
  output[0]  =  1; output[1]  =  2; output[2]  =  3; 
  stencil[0] =  1; stencil[1] =  0; stencil[2] =  1;
  result[0]  = -1; result[1]  =  2; result[2]  = -3;
  
  iter = thrust::transform_if(input.begin(), input.end(),
                              stencil.begin(),
                              output.begin(),
                              thrust::negate<T>(),
                              thrust::identity<T>());
  
  ASSERT_EQUAL(iter - output.begin(), input.size());
  ASSERT_EQUAL(output, result);
}
DECLARE_UNITTEST(TestTransformIfUnaryDeviceSeq);


template<typename Iterator1, typename Iterator2, typename Iterator3, typename Function, typename Iterator4>
__global__
void transform_kernel(Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator3 result1, Function f, Iterator4 result2)
{
  *result2 = thrust::transform(thrust::seq, first1, last1, first2, result1, f);
}


void TestTransformBinaryDeviceSeq()
{
  typedef thrust::device_vector<int> Vector;
  typedef typename Vector::value_type T;
  
  typename Vector::iterator iter;
  
  Vector input1(3);
  Vector input2(3);
  Vector output(3);
  Vector result(3);
  input1[0] =  1; input1[1] = -2; input1[2] =  3;
  input2[0] = -4; input2[1] =  5; input2[2] =  6;
  result[0] =  5; result[1] = -7; result[2] = -3;

  thrust::device_vector<typename Vector::iterator> iter_vec(1);
  
  transform_kernel<<<1,1>>>(input1.begin(), input1.end(), input2.begin(), output.begin(), thrust::minus<T>(), iter_vec.begin());
  iter = iter_vec[0];
  
  ASSERT_EQUAL(iter - output.begin(), input1.size());
  ASSERT_EQUAL(output, result);
}
DECLARE_UNITTEST(TestTransformBinaryDeviceSeq);


template<typename Iterator1, typename Iterator2, typename Iterator3, typename Iterator4, typename Function, typename Predicate, typename Iterator5>
__global__
void transform_if_kernel(Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator3 stencil_first, Iterator4 result1, Function f, Predicate pred, Iterator5 result2)
{
  *result2 = thrust::transform_if(thrust::seq, first1, last1, first2, stencil_first, result1, f, pred);
}


void TestTransformIfBinaryDeviceSeq()
{
  typedef thrust::device_vector<int> Vector;
  typedef typename Vector::value_type T;
  
  typename Vector::iterator iter;
  
  Vector input1(3);
  Vector input2(3);
  Vector stencil(3);
  Vector output(3);
  Vector result(3);
  
  input1[0]  =  1; input1[1]  = -2; input1[2]  =  3;
  input2[0]  = -4; input2[1]  =  5; input2[2]  =  6;
  stencil[0] =  0; stencil[1] =  1; stencil[2] =  0;
  output[0]  =  1; output[1]  =  2; output[2]  =  3;
  result[0]  =  5; result[1]  =  2; result[2]  = -3;
  
  thrust::identity<T> identity;
  
  iter = thrust::transform_if(input1.begin(), input1.end(),
                              input2.begin(),
                              stencil.begin(),
                              output.begin(),
                              thrust::minus<T>(),
                              thrust::not1(identity));
  
  ASSERT_EQUAL(iter - output.begin(), input1.size());
  ASSERT_EQUAL(output, result);
}
DECLARE_UNITTEST(TestTransformIfBinaryDeviceSeq);

