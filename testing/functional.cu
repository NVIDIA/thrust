#include <thrusttest/unittest.h>
#include <thrust/functional.h>
#include <thrust/transform.h>

#include <functional>
#include <algorithm>
    
const size_t NUM_SAMPLES = 10000;

template <class InputVector, class OutputVector, class Operator, class ReferenceOperator>
void TestUnaryFunctional(void)
{
    typedef typename InputVector::value_type  InputType;
    typedef typename OutputVector::value_type OutputType;
    
    thrust::host_vector<InputType>  std_input = thrusttest::random_samples<InputType>(NUM_SAMPLES);
    thrust::host_vector<OutputType> std_output(NUM_SAMPLES);

    InputVector  input = std_input;
    OutputVector output(NUM_SAMPLES);

    thrust::transform(    input.begin(),     input.end(),     output.begin(),          Operator());
    thrust::transform(std_input.begin(), std_input.end(), std_output.begin(), ReferenceOperator());

    ASSERT_EQUAL(output, std_output);
}

template <class InputVector, class OutputVector, class Operator, class ReferenceOperator>
void TestBinaryFunctional(void)
{
    typedef typename InputVector::value_type  InputType;
    typedef typename OutputVector::value_type OutputType;
    
    thrust::host_vector<InputType>  std_input1 = thrusttest::random_samples<InputType>(NUM_SAMPLES);
    thrust::host_vector<InputType>  std_input2 = thrusttest::random_samples<InputType>(NUM_SAMPLES);
    thrust::host_vector<OutputType> std_output(NUM_SAMPLES);

    // Replace zeros to avoid divide by zero exceptions
    std::replace(std_input2.begin(), std_input2.end(), (InputType) 0, (InputType) 1);

    InputVector input1 = std_input1; 
    InputVector input2 = std_input2; 
    OutputVector output(NUM_SAMPLES);

    thrust::transform(    input1.begin(),     input1.end(),      input2.begin(),     output.begin(),          Operator());
    thrust::transform(std_input1.begin(), std_input1.end(),  std_input2.begin(), std_output.begin(), ReferenceOperator());

    // Note: FP division is not bit-equal, even when nvcc is invoked with --prec-div
    ASSERT_ALMOST_EQUAL(output, std_output);
}



// XXX add bool to list
// Instantiate a macro for all integer-like data types
#define INSTANTIATE_INTEGER_TYPES(Macro, vector_type, operator_name)   \
Macro(vector_type, operator_name, char          )                      \
Macro(vector_type, operator_name, unsigned char )                      \
Macro(vector_type, operator_name, short         )                      \
Macro(vector_type, operator_name, unsigned short)                      \
Macro(vector_type, operator_name, int           )                      \
Macro(vector_type, operator_name, unsigned int  )                      \
Macro(vector_type, operator_name, long          )                      \
Macro(vector_type, operator_name, unsigned long )

// Instantiate a macro for all integer and floating point data types
#define INSTANTIATE_ALL_TYPES(Macro, vector_type, operator_name)       \
INSTANTIATE_INTEGER_TYPES(Macro, vector_type, operator_name)           \
Macro(vector_type, operator_name, float)


// op(T) -> T
#define INSTANTIATE_UNARY_ARITHMETIC_FUNCTIONAL_TEST(vector_type, operator_name, data_type) \
    TestUnaryFunctional< thrust::vector_type<data_type>,                                   \
                         thrust::vector_type<data_type>,                                   \
                         thrust::operator_name<data_type>,                                 \
                         std::operator_name<data_type> >();
// XXX revert OutputVector<T> back to bool
// op(T) -> bool
#define INSTANTIATE_UNARY_LOGICAL_FUNCTIONAL_TEST(vector_type, operator_name, data_type) \
    TestUnaryFunctional< thrust::vector_type<data_type>,                                \
                         thrust::vector_type<data_type>,                                \
                         thrust::operator_name<data_type>,                              \
                         std::operator_name<data_type> >();
// op(T,T) -> T
#define INSTANTIATE_BINARY_ARITHMETIC_FUNCTIONAL_TEST(vector_type, operator_name, data_type) \
    TestBinaryFunctional< thrust::vector_type<data_type>,                                   \
                          thrust::vector_type<data_type>,                                   \
                          thrust::operator_name<data_type>,                                 \
                          std::operator_name<data_type> >();
// XXX revert OutputVector<T> back to bool
// op(T,T) -> bool
#define INSTANTIATE_BINARY_LOGICAL_FUNCTIONAL_TEST(vector_type, operator_name, data_type) \
    TestBinaryFunctional< thrust::vector_type<data_type>,                                \
                          thrust::vector_type<data_type>,                                \
                          thrust::operator_name<data_type>,                              \
                          std::operator_name<data_type> >();




// op(T) -> T
#define DECLARE_UNARY_ARITHMETIC_FUNCTIONAL_UNITTEST(operator_name, OperatorName)                          \
void Test##OperatorName##FunctionalHost(void)                                                              \
{                                                                                                          \
    INSTANTIATE_ALL_TYPES( INSTANTIATE_UNARY_ARITHMETIC_FUNCTIONAL_TEST, host_vector,   operator_name);    \
}                                                                                                          \
DECLARE_UNITTEST(Test##OperatorName##FunctionalHost);                                                      \
void Test##OperatorName##FunctionalDevice(void)                                                            \
{                                                                                                          \
    INSTANTIATE_ALL_TYPES( INSTANTIATE_UNARY_ARITHMETIC_FUNCTIONAL_TEST, device_vector, operator_name);    \
}                                                                                                          \
DECLARE_UNITTEST(Test##OperatorName##FunctionalDevice);

// op(T) -> bool
#define DECLARE_UNARY_LOGICAL_FUNCTIONAL_UNITTEST(operator_name, OperatorName)                             \
void Test##OperatorName##FunctionalHost(void)                                                              \
{                                                                                                          \
    INSTANTIATE_ALL_TYPES( INSTANTIATE_UNARY_LOGICAL_FUNCTIONAL_TEST, host_vector,   operator_name);       \
}                                                                                                          \
DECLARE_UNITTEST(Test##OperatorName##FunctionalHost);                                                      \
void Test##OperatorName##FunctionalDevice(void)                                                            \
{                                                                                                          \
    INSTANTIATE_ALL_TYPES( INSTANTIATE_UNARY_LOGICAL_FUNCTIONAL_TEST, device_vector, operator_name);       \
}                                                                                                          \
DECLARE_UNITTEST(Test##OperatorName##FunctionalDevice);

// op(T,T) -> T
#define DECLARE_BINARY_ARITHMETIC_FUNCTIONAL_UNITTEST(operator_name, OperatorName)                         \
void Test##OperatorName##FunctionalHost(void)                                                              \
{                                                                                                          \
    INSTANTIATE_ALL_TYPES( INSTANTIATE_BINARY_ARITHMETIC_FUNCTIONAL_TEST, host_vector,   operator_name);   \
}                                                                                                          \
DECLARE_UNITTEST(Test##OperatorName##FunctionalHost);                                                      \
void Test##OperatorName##FunctionalDevice(void)                                                            \
{                                                                                                          \
    INSTANTIATE_ALL_TYPES( INSTANTIATE_BINARY_ARITHMETIC_FUNCTIONAL_TEST, device_vector, operator_name);   \
}                                                                                                          \
DECLARE_UNITTEST(Test##OperatorName##FunctionalDevice);

// op(T,T) -> T (for integer T only)
#define DECLARE_BINARY_INTEGER_ARITHMETIC_FUNCTIONAL_UNITTEST(operator_name, OperatorName)                     \
void Test##OperatorName##FunctionalHost(void)                                                                  \
{                                                                                                              \
    INSTANTIATE_INTEGER_TYPES( INSTANTIATE_BINARY_ARITHMETIC_FUNCTIONAL_TEST, host_vector,   operator_name);   \
}                                                                                                              \
DECLARE_UNITTEST(Test##OperatorName##FunctionalHost);                                                          \
void Test##OperatorName##FunctionalDevice(void)                                                                \
{                                                                                                              \
    INSTANTIATE_INTEGER_TYPES( INSTANTIATE_BINARY_ARITHMETIC_FUNCTIONAL_TEST, device_vector, operator_name);   \
}                                                                                                              \
DECLARE_UNITTEST(Test##OperatorName##FunctionalDevice);

// op(T,T) -> bool
#define DECLARE_BINARY_LOGICAL_FUNCTIONAL_UNITTEST(operator_name, OperatorName)                         \
void Test##OperatorName##FunctionalHost(void)                                                           \
{                                                                                                       \
    INSTANTIATE_ALL_TYPES( INSTANTIATE_BINARY_LOGICAL_FUNCTIONAL_TEST, host_vector,   operator_name);   \
}                                                                                                       \
DECLARE_UNITTEST(Test##OperatorName##FunctionalHost);                                                   \
void Test##OperatorName##FunctionalDevice(void)                                                         \
{                                                                                                       \
    INSTANTIATE_ALL_TYPES( INSTANTIATE_BINARY_LOGICAL_FUNCTIONAL_TEST, device_vector, operator_name);   \
}                                                                                                       \
DECLARE_UNITTEST(Test##OperatorName##FunctionalDevice);




// Create the unit tests
DECLARE_UNARY_ARITHMETIC_FUNCTIONAL_UNITTEST(negate, Negate);
DECLARE_UNARY_LOGICAL_FUNCTIONAL_UNITTEST(logical_not, LogicalNot);

DECLARE_BINARY_ARITHMETIC_FUNCTIONAL_UNITTEST(plus,       Plus      );
DECLARE_BINARY_ARITHMETIC_FUNCTIONAL_UNITTEST(minus,      Minus     );
DECLARE_BINARY_ARITHMETIC_FUNCTIONAL_UNITTEST(multiplies, Multiplies);
DECLARE_BINARY_ARITHMETIC_FUNCTIONAL_UNITTEST(divides,    Divides   );

DECLARE_BINARY_INTEGER_ARITHMETIC_FUNCTIONAL_UNITTEST(modulus, Modulus);

DECLARE_BINARY_LOGICAL_FUNCTIONAL_UNITTEST(equal_to,      EqualTo     );
DECLARE_BINARY_LOGICAL_FUNCTIONAL_UNITTEST(not_equal_to,  NotEqualTo  );
DECLARE_BINARY_LOGICAL_FUNCTIONAL_UNITTEST(greater,       Greater     );
DECLARE_BINARY_LOGICAL_FUNCTIONAL_UNITTEST(less,          Less        );
DECLARE_BINARY_LOGICAL_FUNCTIONAL_UNITTEST(greater_equal, GreaterEqual);
DECLARE_BINARY_LOGICAL_FUNCTIONAL_UNITTEST(less_equal,    LessEqual   );
DECLARE_BINARY_LOGICAL_FUNCTIONAL_UNITTEST(logical_and,   LogicalAnd  );
DECLARE_BINARY_LOGICAL_FUNCTIONAL_UNITTEST(logical_or,    LogicalOr   );


// Ad-hoc testing for other functionals
template <class Vector>
void TestIdentityFunctional(void)
{
    typedef typename Vector::value_type T;

    Vector input(3);
    input[0] = 0; input[1] = 1; input[2] = 2;

    Vector output(3);

    thrust::transform(input.begin(), input.end(), output.begin(), thrust::identity<T>());

    ASSERT_EQUAL(input, output);
}
DECLARE_VECTOR_UNITTEST(TestIdentityFunctional);



template <class Vector>
void TestMaximumFunctional(void)
{
    typedef typename Vector::value_type T;

    Vector input1(3);
    Vector input2(3);
    input1[0] = 8; input1[1] = 3; input1[2] = 7;
    input2[0] = 5; input2[1] = 6; input2[2] = 9;

    Vector output(3);

    thrust::transform(input1.begin(), input1.end(), 
                       input2.begin(), 
                       output.begin(), 
                       thrust::maximum<T>());

    ASSERT_EQUAL(output[0], 8);
    ASSERT_EQUAL(output[1], 6);
    ASSERT_EQUAL(output[2], 9);
}
DECLARE_VECTOR_UNITTEST(TestMaximumFunctional);

template <class Vector>
void TestMinimumFunctional(void)
{
    typedef typename Vector::value_type T;

    Vector input1(3);
    Vector input2(3);
    input1[0] = 8; input1[1] = 3; input1[2] = 7;
    input2[0] = 5; input2[1] = 6; input2[2] = 9;

    Vector output(3);

    thrust::transform(input1.begin(), input1.end(), 
                       input2.begin(), 
                       output.begin(), 
                       thrust::minimum<T>());

    ASSERT_EQUAL(output[0], 5);
    ASSERT_EQUAL(output[1], 3);
    ASSERT_EQUAL(output[2], 7);
}
DECLARE_VECTOR_UNITTEST(TestMinimumFunctional);

