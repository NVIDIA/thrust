#include <thrusttest/unittest.h>
#include <thrust/transform.h>

template <class Vector>
void TestTransformUnarySimple(void)
{
    typedef typename Vector::value_type T;

    Vector input(3);
    Vector output(3);
    Vector result(3);
    input[0]  =  1; input[1]  = -2; input[2]  =  3;
    result[0] = -1; result[1] =  2; result[2] = -3;

    thrust::transform(input.begin(), input.end(), output.begin(), thrust::negate<T>());
    
    ASSERT_EQUAL(output, result);
}
DECLARE_VECTOR_UNITTEST(TestTransformUnarySimple);


template <class Vector>
void TestTransformIfUnarySimple(void)
{
    typedef typename Vector::value_type T;

    Vector input(3);
    Vector stencil(3);
    Vector output(3);
    Vector result(3);

    input[0]   =  1; input[1]   = -2; input[2]   =  3;
    output[0]  =  1; output[1]  =  2; output[2]  =  3; 
    stencil[0] =  1; stencil[1] =  0; stencil[2] =  1;
    result[0]  = -1; result[1]  =  2; result[2]  = -3;

    thrust::transform_if(input.begin(), input.end(),
                          stencil.begin(),
                          output.begin(),
                          thrust::negate<T>(),
                          thrust::identity<T>());
    
    ASSERT_EQUAL(output, result);
}
DECLARE_VECTOR_UNITTEST(TestTransformIfUnarySimple);


template <class Vector>
void TestTransformBinarySimple(void)
{
    typedef typename Vector::value_type T;

    Vector input1(3);
    Vector input2(3);
    Vector output(3);
    Vector result(3);
    input1[0] =  1; input1[1] = -2; input1[2] =  3;
    input2[0] = -4; input2[1] =  5; input2[2] =  6;
    result[0] =  5; result[1] = -7; result[2] = -3;

    thrust::transform(input1.begin(), input1.end(), input2.begin(), output.begin(), thrust::minus<T>());
    
    ASSERT_EQUAL(output, result);
}
DECLARE_VECTOR_UNITTEST(TestTransformBinarySimple);


template <class Vector>
void TestTransformIfBinarySimple(void)
{
    typedef typename Vector::value_type T;

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

    thrust::transform_if(input1.begin(), input1.end(),
                          input2.begin(),
                          stencil.begin(),
                          output.begin(),
                          thrust::minus<T>(),
                          thrust::not1(identity));
    
    ASSERT_EQUAL(output, result);
}
DECLARE_VECTOR_UNITTEST(TestTransformIfBinarySimple);


template <typename T>
void TestTransformUnary(const size_t n)
{
    thrust::host_vector<T>   h_input = thrusttest::random_integers<T>(n);
    thrust::device_vector<T> d_input = h_input;

    thrust::host_vector<T>   h_output(n);
    thrust::device_vector<T> d_output(n);

    thrust::transform(h_input.begin(), h_input.end(), h_output.begin(), thrust::negate<T>());
    thrust::transform(d_input.begin(), d_input.end(), d_output.begin(), thrust::negate<T>());
    
    ASSERT_EQUAL(h_output, d_output);
}
DECLARE_VARIABLE_UNITTEST(TestTransformUnary);


struct is_positive
{
  template<typename T>
  __host__ __device__
  bool operator()(T &x)
  {
    return x > 0;
  } // end operator()()
}; // end is_positive


template <typename T>
void TestTransformIfUnary(const size_t n)
{
    thrust::host_vector<T>   h_input   = thrusttest::random_integers<T>(n);
    thrust::host_vector<T>   h_stencil = thrusttest::random_integers<T>(n);
    thrust::host_vector<T>   h_output  = thrusttest::random_integers<T>(n);

    thrust::device_vector<T> d_input   = h_input;
    thrust::device_vector<T> d_stencil = h_stencil;
    thrust::device_vector<T> d_output  = h_output;

    thrust::transform_if(h_input.begin(), h_input.end(),
                          h_stencil.begin(),
                          h_output.begin(),
                          thrust::negate<T>(), is_positive());

    thrust::transform_if(d_input.begin(), d_input.end(),
                          d_stencil.begin(),
                          d_output.begin(),
                          thrust::negate<T>(), is_positive());
    
    ASSERT_EQUAL(h_output, d_output);
}
DECLARE_VARIABLE_UNITTEST(TestTransformIfUnary);


template <typename T>
void TestTransformBinary(const size_t n)
{
    thrust::host_vector<T>   h_input1 = thrusttest::random_integers<T>(n);
    thrust::host_vector<T>   h_input2 = thrusttest::random_integers<T>(n);
    thrust::device_vector<T> d_input1 = h_input1;
    thrust::device_vector<T> d_input2 = h_input2;

    thrust::host_vector<T>   h_output(n);
    thrust::device_vector<T> d_output(n);

    thrust::transform(h_input1.begin(), h_input1.end(), h_input2.begin(), h_output.begin(), thrust::minus<T>());
    thrust::transform(d_input1.begin(), d_input1.end(), d_input2.begin(), d_output.begin(), thrust::minus<T>());
    
    ASSERT_EQUAL(h_output, d_output);
    
    thrust::transform(h_input1.begin(), h_input1.end(), h_input2.begin(), h_output.begin(), thrust::multiplies<T>());
    thrust::transform(d_input1.begin(), d_input1.end(), d_input2.begin(), d_output.begin(), thrust::multiplies<T>());
    
    ASSERT_EQUAL(h_output, d_output);
}
DECLARE_VARIABLE_UNITTEST(TestTransformBinary);


template <typename T>
void TestTransformIfBinary(const size_t n)
{
    thrust::host_vector<T>   h_input1  = thrusttest::random_integers<T>(n);
    thrust::host_vector<T>   h_input2  = thrusttest::random_integers<T>(n);
    thrust::host_vector<T>   h_stencil = thrusttest::random_integers<T>(n);
    thrust::host_vector<T>   h_output  = thrusttest::random_integers<T>(n);

    thrust::device_vector<T> d_input1  = h_input1;
    thrust::device_vector<T> d_input2  = h_input2;
    thrust::device_vector<T> d_stencil = h_stencil;
    thrust::device_vector<T> d_output  = h_output;

    thrust::transform_if(h_input1.begin(), h_input1.end(),
                         h_input2.begin(),
                         h_stencil.begin(),
                         h_output.begin(),
                         thrust::minus<T>(), is_positive());

    thrust::transform_if(d_input1.begin(), d_input1.end(),
                         d_input2.begin(),
                         d_stencil.begin(),
                         d_output.begin(),
                         thrust::minus<T>(), is_positive());
    
    ASSERT_EQUAL(h_output, d_output);

    h_stencil = thrusttest::random_integers<T>(n);
    d_stencil = h_stencil;
    
    thrust::transform_if(h_input1.begin(), h_input1.end(),
                         h_input2.begin(),
                         h_stencil.begin(),
                         h_output.begin(),
                         thrust::multiplies<T>(), is_positive());

    thrust::transform_if(d_input1.begin(), d_input1.end(),
                         d_input2.begin(),
                         d_stencil.begin(),
                         d_output.begin(),
                         thrust::multiplies<T>(), is_positive());
    
    ASSERT_EQUAL(h_output, d_output);
}
DECLARE_VARIABLE_UNITTEST(TestTransformIfBinary);

