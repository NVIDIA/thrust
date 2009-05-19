#include <komradetest/unittest.h>
#include <komrade/transform.h>

template <class Vector>
void TestTransformUnarySimple(void)
{
    typedef typename Vector::value_type T;

    Vector input(3);
    Vector output(3);
    Vector result(3);
    input[0]  =  1; input[1]  = -2; input[2]  =  3;
    result[0] = -1; result[1] =  2; result[2] = -3;

    komrade::transform(input.begin(), input.end(), output.begin(), komrade::negate<T>());
    
    ASSERT_EQUAL(output, result);
}
DECLARE_VECTOR_UNITTEST(TestTransformUnarySimple);


template <class Vector>
void TestPredicatedTransformUnarySimple(void)
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

    komrade::experimental::predicated_transform(input.begin(), input.end(),
                                                stencil.begin(),
                                                output.begin(),
                                                komrade::negate<T>(),
                                                komrade::identity<T>());
    
    ASSERT_EQUAL(output, result);
}
DECLARE_VECTOR_UNITTEST(TestPredicatedTransformUnarySimple);


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

    komrade::transform(input1.begin(), input1.end(), input2.begin(), output.begin(), komrade::minus<T>());
    
    ASSERT_EQUAL(output, result);
}
DECLARE_VECTOR_UNITTEST(TestTransformBinarySimple);


template <class Vector>
void TestPredicatedTransformBinarySimple(void)
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

    komrade::identity<T> identity;

    komrade::experimental::predicated_transform(input1.begin(), input1.end(),
                                                input2.begin(),
                                                stencil.begin(),
                                                output.begin(),
                                                komrade::minus<T>(),
                                                komrade::not1(identity));
    
    ASSERT_EQUAL(output, result);
}
DECLARE_VECTOR_UNITTEST(TestPredicatedTransformBinarySimple);


template <typename T>
void TestTransformUnary(const size_t n)
{
    komrade::host_vector<T>   h_input = komradetest::random_integers<T>(n);
    komrade::device_vector<T> d_input = h_input;

    komrade::host_vector<T>   h_output(n);
    komrade::device_vector<T> d_output(n);

    komrade::transform(h_input.begin(), h_input.end(), h_output.begin(), komrade::negate<T>());
    komrade::transform(d_input.begin(), d_input.end(), d_output.begin(), komrade::negate<T>());
    
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
void TestPredicatedTransformUnary(const size_t n)
{
    komrade::host_vector<T>   h_input   = komradetest::random_integers<T>(n);
    komrade::host_vector<T>   h_stencil = komradetest::random_integers<T>(n);
    komrade::host_vector<T>   h_output  = komradetest::random_integers<T>(n);

    komrade::device_vector<T> d_input   = h_input;
    komrade::device_vector<T> d_stencil = h_stencil;
    komrade::device_vector<T> d_output  = h_output;

    komrade::experimental::predicated_transform(h_input.begin(), h_input.end(),
                                                h_stencil.begin(),
                                                h_output.begin(),
                                                komrade::negate<T>(), is_positive());

    komrade::experimental::predicated_transform(d_input.begin(), d_input.end(),
                                                d_stencil.begin(),
                                                d_output.begin(),
                                                komrade::negate<T>(), is_positive());
    
    ASSERT_EQUAL(h_output, d_output);
}
DECLARE_VARIABLE_UNITTEST(TestPredicatedTransformUnary);


template <typename T>
void TestTransformBinary(const size_t n)
{
    komrade::host_vector<T>   h_input1 = komradetest::random_integers<T>(n);
    komrade::host_vector<T>   h_input2 = komradetest::random_integers<T>(n);
    komrade::device_vector<T> d_input1 = h_input1;
    komrade::device_vector<T> d_input2 = h_input2;

    komrade::host_vector<T>   h_output(n);
    komrade::device_vector<T> d_output(n);

    komrade::transform(h_input1.begin(), h_input1.end(), h_input2.begin(), h_output.begin(), komrade::minus<T>());
    komrade::transform(d_input1.begin(), d_input1.end(), d_input2.begin(), d_output.begin(), komrade::minus<T>());
    
    ASSERT_EQUAL(h_output, d_output);
    
    komrade::transform(h_input1.begin(), h_input1.end(), h_input2.begin(), h_output.begin(), komrade::multiplies<T>());
    komrade::transform(d_input1.begin(), d_input1.end(), d_input2.begin(), d_output.begin(), komrade::multiplies<T>());
    
    ASSERT_EQUAL(h_output, d_output);
}
DECLARE_VARIABLE_UNITTEST(TestTransformBinary);


template <typename T>
void TestPredicatedTransformBinary(const size_t n)
{
    komrade::host_vector<T>   h_input1  = komradetest::random_integers<T>(n);
    komrade::host_vector<T>   h_input2  = komradetest::random_integers<T>(n);
    komrade::host_vector<T>   h_stencil = komradetest::random_integers<T>(n);
    komrade::host_vector<T>   h_output  = komradetest::random_integers<T>(n);

    komrade::device_vector<T> d_input1  = h_input1;
    komrade::device_vector<T> d_input2  = h_input2;
    komrade::device_vector<T> d_stencil = h_stencil;
    komrade::device_vector<T> d_output  = h_output;

    komrade::experimental::predicated_transform(h_input1.begin(), h_input1.end(),
                                                h_input2.begin(),
                                                h_stencil.begin(),
                                                h_output.begin(),
                                                komrade::minus<T>(), is_positive());

    komrade::experimental::predicated_transform(d_input1.begin(), d_input1.end(),
                                                d_input2.begin(),
                                                d_stencil.begin(),
                                                d_output.begin(),
                                                komrade::minus<T>(), is_positive());
    
    ASSERT_EQUAL(h_output, d_output);

    h_stencil = komradetest::random_integers<T>(n);
    d_stencil = h_stencil;
    
    komrade::experimental::predicated_transform(h_input1.begin(), h_input1.end(),
                                                h_input2.begin(),
                                                h_stencil.begin(),
                                                h_output.begin(),
                                                komrade::multiplies<T>(), is_positive());

    komrade::experimental::predicated_transform(d_input1.begin(), d_input1.end(),
                                                d_input2.begin(),
                                                d_stencil.begin(),
                                                d_output.begin(),
                                                komrade::multiplies<T>(), is_positive());
    
    ASSERT_EQUAL(h_output, d_output);
}
DECLARE_VARIABLE_UNITTEST(TestPredicatedTransformBinary);

