#include <unittest/unittest.h>
#include <thrust/range/algorithm/transform.h>
#include <thrust/iterator/counting_iterator.h>

template <class Vector>
void TestRangeTransformUnarySimple(void)
{
    typedef typename Vector::value_type T;

    Vector input(3);
    Vector output(3);
    Vector result(3);
    input[0]  =  1; input[1]  = -2; input[2]  =  3;
    result[0] = -1; result[1] =  2; result[2] = -3;

    size_t result_size = thrust::experimental::range::transform(input, output, thrust::negate<T>()).size();
    
    ASSERT_EQUAL(0, result_size);
    ASSERT_EQUAL(output, result);
}
DECLARE_VECTOR_UNITTEST(TestRangeTransformUnarySimple);


template <class Vector>
void TestRangeTransformIfUnarySimple(void)
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

    size_t result_size = thrust::experimental::range::transform_if(input,
                                                                   stencil,
                                                                   output,
                                                                   thrust::negate<T>(),
                                                                   thrust::identity<T>());
    
    ASSERT_EQUAL(0, result_size);
    ASSERT_EQUAL(output, result);
}
DECLARE_VECTOR_UNITTEST(TestRangeTransformIfUnarySimple);


template <class Vector>
void TestRangeTransformBinarySimple(void)
{
    typedef typename Vector::value_type T;

    Vector input1(3);
    Vector input2(3);
    Vector output(3);
    Vector result(3);
    input1[0] =  1; input1[1] = -2; input1[2] =  3;
    input2[0] = -4; input2[1] =  5; input2[2] =  6;
    result[0] =  5; result[1] = -7; result[2] = -3;

    size_t result_size = thrust::experimental::range::transform(input1, input2, output, thrust::minus<T>());
    
    ASSERT_EQUAL(0, result_size);
    ASSERT_EQUAL(output, result);
}
DECLARE_VECTOR_UNITTEST(TestRangeTransformBinarySimple);


//template <class Vector>
//void TestTransformIfBinarySimple(void)
//{
//    typedef typename Vector::value_type T;
//    
//    typename Vector::iterator iter;
//
//    Vector input1(3);
//    Vector input2(3);
//    Vector stencil(3);
//    Vector output(3);
//    Vector result(3);
//
//    input1[0]  =  1; input1[1]  = -2; input1[2]  =  3;
//    input2[0]  = -4; input2[1]  =  5; input2[2]  =  6;
//    stencil[0] =  0; stencil[1] =  1; stencil[2] =  0;
//    output[0]  =  1; output[1]  =  2; output[2]  =  3;
//    result[0]  =  5; result[1]  =  2; result[2]  = -3;
//
//    thrust::identity<T> identity;
//
//    iter = thrust::transform_if(input1.begin(), input1.end(),
//                                input2.begin(),
//                                stencil.begin(),
//                                output.begin(),
//                                thrust::minus<T>(),
//                                thrust::not1(identity));
//    
//    ASSERT_EQUAL(iter - output.begin(), input1.size());
//    ASSERT_EQUAL(output, result);
//}
//DECLARE_VECTOR_UNITTEST(TestTransformIfBinarySimple);
//
//
//template <typename T>
//void TestTransformUnary(const size_t n)
//{
//    thrust::host_vector<T>   h_input = unittest::random_integers<T>(n);
//    thrust::device_vector<T> d_input = h_input;
//
//    thrust::host_vector<T>   h_output(n);
//    thrust::device_vector<T> d_output(n);
//
//    thrust::transform(h_input.begin(), h_input.end(), h_output.begin(), thrust::negate<T>());
//    thrust::transform(d_input.begin(), d_input.end(), d_output.begin(), thrust::negate<T>());
//    
//    ASSERT_EQUAL(h_output, d_output);
//}
//DECLARE_VARIABLE_UNITTEST(TestTransformUnary);
//
//
//struct is_positive
//{
//  template<typename T>
//  __host__ __device__
//  bool operator()(T &x)
//  {
//    return x > 0;
//  } // end operator()()
//}; // end is_positive
//
//
//template <typename T>
//void TestTransformIfUnary(const size_t n)
//{
//    thrust::host_vector<T>   h_input   = unittest::random_integers<T>(n);
//    thrust::host_vector<T>   h_stencil = unittest::random_integers<T>(n);
//    thrust::host_vector<T>   h_output  = unittest::random_integers<T>(n);
//
//    thrust::device_vector<T> d_input   = h_input;
//    thrust::device_vector<T> d_stencil = h_stencil;
//    thrust::device_vector<T> d_output  = h_output;
//
//    thrust::transform_if(h_input.begin(), h_input.end(),
//                          h_stencil.begin(),
//                          h_output.begin(),
//                          thrust::negate<T>(), is_positive());
//
//    thrust::transform_if(d_input.begin(), d_input.end(),
//                          d_stencil.begin(),
//                          d_output.begin(),
//                          thrust::negate<T>(), is_positive());
//    
//    ASSERT_EQUAL(h_output, d_output);
//}
//DECLARE_VARIABLE_UNITTEST(TestTransformIfUnary);
//
//
//template <typename T>
//void TestTransformBinary(const size_t n)
//{
//    thrust::host_vector<T>   h_input1 = unittest::random_integers<T>(n);
//    thrust::host_vector<T>   h_input2 = unittest::random_integers<T>(n);
//    thrust::device_vector<T> d_input1 = h_input1;
//    thrust::device_vector<T> d_input2 = h_input2;
//
//    thrust::host_vector<T>   h_output(n);
//    thrust::device_vector<T> d_output(n);
//
//    thrust::transform(h_input1.begin(), h_input1.end(), h_input2.begin(), h_output.begin(), thrust::minus<T>());
//    thrust::transform(d_input1.begin(), d_input1.end(), d_input2.begin(), d_output.begin(), thrust::minus<T>());
//    
//    ASSERT_EQUAL(h_output, d_output);
//    
//    thrust::transform(h_input1.begin(), h_input1.end(), h_input2.begin(), h_output.begin(), thrust::multiplies<T>());
//    thrust::transform(d_input1.begin(), d_input1.end(), d_input2.begin(), d_output.begin(), thrust::multiplies<T>());
//    
//    ASSERT_EQUAL(h_output, d_output);
//}
//DECLARE_VARIABLE_UNITTEST(TestTransformBinary);
//
//
//template <typename T>
//void TestTransformIfBinary(const size_t n)
//{
//    thrust::host_vector<T>   h_input1  = unittest::random_integers<T>(n);
//    thrust::host_vector<T>   h_input2  = unittest::random_integers<T>(n);
//    thrust::host_vector<T>   h_stencil = unittest::random_integers<T>(n);
//    thrust::host_vector<T>   h_output  = unittest::random_integers<T>(n);
//
//    thrust::device_vector<T> d_input1  = h_input1;
//    thrust::device_vector<T> d_input2  = h_input2;
//    thrust::device_vector<T> d_stencil = h_stencil;
//    thrust::device_vector<T> d_output  = h_output;
//
//    thrust::transform_if(h_input1.begin(), h_input1.end(),
//                         h_input2.begin(),
//                         h_stencil.begin(),
//                         h_output.begin(),
//                         thrust::minus<T>(), is_positive());
//
//    thrust::transform_if(d_input1.begin(), d_input1.end(),
//                         d_input2.begin(),
//                         d_stencil.begin(),
//                         d_output.begin(),
//                         thrust::minus<T>(), is_positive());
//    
//    ASSERT_EQUAL(h_output, d_output);
//
//    h_stencil = unittest::random_integers<T>(n);
//    d_stencil = h_stencil;
//    
//    thrust::transform_if(h_input1.begin(), h_input1.end(),
//                         h_input2.begin(),
//                         h_stencil.begin(),
//                         h_output.begin(),
//                         thrust::multiplies<T>(), is_positive());
//
//    thrust::transform_if(d_input1.begin(), d_input1.end(),
//                         d_input2.begin(),
//                         d_stencil.begin(),
//                         d_output.begin(),
//                         thrust::multiplies<T>(), is_positive());
//    
//    ASSERT_EQUAL(h_output, d_output);
//}
//DECLARE_VARIABLE_UNITTEST(TestTransformIfBinary);
//
//
//template <class Vector>
//void TestTransformUnaryCountingIterator(void)
//{
//    typedef typename Vector::value_type T;
//
//    thrust::counting_iterator<T> first(1);
//
//    Vector output(3);
//
//    thrust::transform(first, first + 3, output.begin(), thrust::identity<T>());
//    
//    Vector result(3);
//    result[0] = 1; result[1] = 2; result[2] = 3;
//
//    ASSERT_EQUAL(output, result);
//}
//DECLARE_VECTOR_UNITTEST(TestTransformUnaryCountingIterator);
//
//template <class Vector>
//void TestTransformBinaryCountingIterator(void)
//{
//    typedef typename Vector::value_type T;
//
//    thrust::counting_iterator<T> first(1);
//
//    Vector output(3);
//
//    thrust::transform(first, first + 3, first, output.begin(), thrust::plus<T>());
//    
//    Vector result(3);
//    result[0] = 2; result[1] = 4; result[2] = 6;
//
//    ASSERT_EQUAL(output, result);
//}
//DECLARE_VECTOR_UNITTEST(TestTransformBinaryCountingIterator);
//
//
//template <typename T>
//struct plus_mod3 : public thrust::binary_function<T,T,T>
//{
//    T * table;
//
//    plus_mod3(T * table) : table(table) {}
//
//    __host__ __device__
//    T operator()(T a, T b)
//    {
//        return table[(int) (a + b)];
//    }
//};
//
//template <typename Vector>
//void TestTransformWithIndirection(void)
//{
//    // add numbers modulo 3 with external lookup table
//    typedef typename Vector::value_type T;
//
//    Vector input1(7);
//    Vector input2(7);
//    Vector output(7, 0);
//    input1[0] = 0;  input2[0] = 2; 
//    input1[1] = 1;  input2[1] = 2;
//    input1[2] = 2;  input2[2] = 2;
//    input1[3] = 1;  input2[3] = 0;
//    input1[4] = 2;  input2[4] = 2;
//    input1[5] = 0;  input2[5] = 1;
//    input1[6] = 1;  input2[6] = 0;
//
//    Vector table(6);
//    table[0] = 0;
//    table[1] = 1;
//    table[2] = 2;
//    table[3] = 0;
//    table[4] = 1;
//    table[5] = 2;
//
//    thrust::transform(input1.begin(), input1.end(),
//                      input2.begin(), 
//                      output.begin(),
//                      plus_mod3<T>(thrust::raw_pointer_cast(&table[0])));
//    
//    ASSERT_EQUAL(output[0], T(2));
//    ASSERT_EQUAL(output[1], T(0));
//    ASSERT_EQUAL(output[2], T(1));
//    ASSERT_EQUAL(output[3], T(1));
//    ASSERT_EQUAL(output[4], T(1));
//    ASSERT_EQUAL(output[5], T(1));
//    ASSERT_EQUAL(output[6], T(1));
//}
//DECLARE_VECTOR_UNITTEST(TestTransformWithIndirection);
//
