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

