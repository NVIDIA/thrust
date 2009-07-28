#include <thrusttest/unittest.h>
#include <thrust/binary_search.h>

#include <thrust/sequence.h>
#include <thrust/sort.h>

//////////////////////
// Scalar Functions //
//////////////////////

template <class Vector>
void TestScalarLowerBoundSimple(void)
{
    typedef typename Vector::value_type T;

    Vector vec(5);

    vec[0] = 0;
    vec[1] = 2;
    vec[2] = 5;
    vec[3] = 7;
    vec[4] = 8;

    ASSERT_EQUAL(thrust::lower_bound(vec.begin(), vec.end(), 0) - vec.begin(), 0);
    ASSERT_EQUAL(thrust::lower_bound(vec.begin(), vec.end(), 1) - vec.begin(), 1);
    ASSERT_EQUAL(thrust::lower_bound(vec.begin(), vec.end(), 2) - vec.begin(), 1);
    ASSERT_EQUAL(thrust::lower_bound(vec.begin(), vec.end(), 3) - vec.begin(), 2);
    ASSERT_EQUAL(thrust::lower_bound(vec.begin(), vec.end(), 4) - vec.begin(), 2);
    ASSERT_EQUAL(thrust::lower_bound(vec.begin(), vec.end(), 5) - vec.begin(), 2);
    ASSERT_EQUAL(thrust::lower_bound(vec.begin(), vec.end(), 6) - vec.begin(), 3);
    ASSERT_EQUAL(thrust::lower_bound(vec.begin(), vec.end(), 7) - vec.begin(), 3);
    ASSERT_EQUAL(thrust::lower_bound(vec.begin(), vec.end(), 8) - vec.begin(), 4);
    ASSERT_EQUAL(thrust::lower_bound(vec.begin(), vec.end(), 9) - vec.begin(), 5);
}
DECLARE_VECTOR_UNITTEST(TestScalarLowerBoundSimple);


template <class Vector>
void TestScalarUpperBoundSimple(void)
{
    typedef typename Vector::value_type T;

    Vector vec(5);

    vec[0] = 0;
    vec[1] = 2;
    vec[2] = 5;
    vec[3] = 7;
    vec[4] = 8;

    ASSERT_EQUAL(thrust::upper_bound(vec.begin(), vec.end(), 0) - vec.begin(), 1);
    ASSERT_EQUAL(thrust::upper_bound(vec.begin(), vec.end(), 1) - vec.begin(), 1);
    ASSERT_EQUAL(thrust::upper_bound(vec.begin(), vec.end(), 2) - vec.begin(), 2);
    ASSERT_EQUAL(thrust::upper_bound(vec.begin(), vec.end(), 3) - vec.begin(), 2);
    ASSERT_EQUAL(thrust::upper_bound(vec.begin(), vec.end(), 4) - vec.begin(), 2);
    ASSERT_EQUAL(thrust::upper_bound(vec.begin(), vec.end(), 5) - vec.begin(), 3);
    ASSERT_EQUAL(thrust::upper_bound(vec.begin(), vec.end(), 6) - vec.begin(), 3);
    ASSERT_EQUAL(thrust::upper_bound(vec.begin(), vec.end(), 7) - vec.begin(), 4);
    ASSERT_EQUAL(thrust::upper_bound(vec.begin(), vec.end(), 8) - vec.begin(), 5);
    ASSERT_EQUAL(thrust::upper_bound(vec.begin(), vec.end(), 9) - vec.begin(), 5);
}
DECLARE_VECTOR_UNITTEST(TestScalarUpperBoundSimple);


template <class Vector>
void TestScalarBinarySearchSimple(void)
{
    typedef typename Vector::value_type T;

    Vector vec(5);

    vec[0] = 0;
    vec[1] = 2;
    vec[2] = 5;
    vec[3] = 7;
    vec[4] = 8;

    ASSERT_EQUAL(thrust::binary_search(vec.begin(), vec.end(), 0),  true);
    ASSERT_EQUAL(thrust::binary_search(vec.begin(), vec.end(), 1), false);
    ASSERT_EQUAL(thrust::binary_search(vec.begin(), vec.end(), 2),  true);
    ASSERT_EQUAL(thrust::binary_search(vec.begin(), vec.end(), 3), false);
    ASSERT_EQUAL(thrust::binary_search(vec.begin(), vec.end(), 4), false);
    ASSERT_EQUAL(thrust::binary_search(vec.begin(), vec.end(), 5),  true);
    ASSERT_EQUAL(thrust::binary_search(vec.begin(), vec.end(), 6), false);
    ASSERT_EQUAL(thrust::binary_search(vec.begin(), vec.end(), 7),  true);
    ASSERT_EQUAL(thrust::binary_search(vec.begin(), vec.end(), 8),  true);
    ASSERT_EQUAL(thrust::binary_search(vec.begin(), vec.end(), 9), false);
}
DECLARE_VECTOR_UNITTEST(TestScalarBinarySearchSimple);


template <class Vector>
void TestScalarEqualRangeSimple(void)
{
    typedef typename Vector::value_type T;

    Vector vec(5);

    vec[0] = 0;
    vec[1] = 2;
    vec[2] = 5;
    vec[3] = 7;
    vec[4] = 8;

    ASSERT_EQUAL(thrust::equal_range(vec.begin(), vec.end(), 0).first - vec.begin(), 0);
    ASSERT_EQUAL(thrust::equal_range(vec.begin(), vec.end(), 1).first - vec.begin(), 1);
    ASSERT_EQUAL(thrust::equal_range(vec.begin(), vec.end(), 2).first - vec.begin(), 1);
    ASSERT_EQUAL(thrust::equal_range(vec.begin(), vec.end(), 3).first - vec.begin(), 2);
    ASSERT_EQUAL(thrust::equal_range(vec.begin(), vec.end(), 4).first - vec.begin(), 2);
    ASSERT_EQUAL(thrust::equal_range(vec.begin(), vec.end(), 5).first - vec.begin(), 2);
    ASSERT_EQUAL(thrust::equal_range(vec.begin(), vec.end(), 6).first - vec.begin(), 3);
    ASSERT_EQUAL(thrust::equal_range(vec.begin(), vec.end(), 7).first - vec.begin(), 3);
    ASSERT_EQUAL(thrust::equal_range(vec.begin(), vec.end(), 8).first - vec.begin(), 4);
    ASSERT_EQUAL(thrust::equal_range(vec.begin(), vec.end(), 9).first - vec.begin(), 5);
    
    ASSERT_EQUAL(thrust::equal_range(vec.begin(), vec.end(), 0).second - vec.begin(), 1);
    ASSERT_EQUAL(thrust::equal_range(vec.begin(), vec.end(), 1).second - vec.begin(), 1);
    ASSERT_EQUAL(thrust::equal_range(vec.begin(), vec.end(), 2).second - vec.begin(), 2);
    ASSERT_EQUAL(thrust::equal_range(vec.begin(), vec.end(), 3).second - vec.begin(), 2);
    ASSERT_EQUAL(thrust::equal_range(vec.begin(), vec.end(), 4).second - vec.begin(), 2);
    ASSERT_EQUAL(thrust::equal_range(vec.begin(), vec.end(), 5).second - vec.begin(), 3);
    ASSERT_EQUAL(thrust::equal_range(vec.begin(), vec.end(), 6).second - vec.begin(), 3);
    ASSERT_EQUAL(thrust::equal_range(vec.begin(), vec.end(), 7).second - vec.begin(), 4);
    ASSERT_EQUAL(thrust::equal_range(vec.begin(), vec.end(), 8).second - vec.begin(), 5);
    ASSERT_EQUAL(thrust::equal_range(vec.begin(), vec.end(), 9).second - vec.begin(), 5);
}
DECLARE_VECTOR_UNITTEST(TestScalarEqualRangeSimple);


//////////////////////
// Vector Functions //
//////////////////////

// convert xxx_vector<T1> to xxx_vector<T2> 
template <class ExampleVector, typename NewType> 
struct vector_like
{
    typedef typename ExampleVector::allocator_type alloc;
    typedef typename alloc::template rebind<NewType>::other new_alloc;
    typedef thrust::detail::vector_base<NewType, new_alloc> type;
};

template <class Vector>
void TestVectorLowerBoundSimple(void)
{
    typedef typename Vector::value_type T;

    Vector vec(5);

    vec[0] = 0;
    vec[1] = 2;
    vec[2] = 5;
    vec[3] = 7;
    vec[4] = 8;

    Vector input(10);
    thrust::sequence(input.begin(), input.end());

    typedef typename vector_like<Vector, int>::type IntVector;

    // test with integral output type
    IntVector integral_output(10);
    thrust::experimental::lower_bound(vec.begin(), vec.end(), input.begin(), input.end(), integral_output.begin());
    
    typename IntVector::iterator output_end = thrust::experimental::lower_bound(vec.begin(), vec.end(), input.begin(), input.end(), integral_output.begin());

    ASSERT_EQUAL((output_end - integral_output.begin()), 10);

    ASSERT_EQUAL(integral_output[0], 0);
    ASSERT_EQUAL(integral_output[1], 1);
    ASSERT_EQUAL(integral_output[2], 1);
    ASSERT_EQUAL(integral_output[3], 2);
    ASSERT_EQUAL(integral_output[4], 2);
    ASSERT_EQUAL(integral_output[5], 2);
    ASSERT_EQUAL(integral_output[6], 3);
    ASSERT_EQUAL(integral_output[7], 3);
    ASSERT_EQUAL(integral_output[8], 4);
    ASSERT_EQUAL(integral_output[9], 5);

//    // test with interator output type
//    typedef typename vector_like<Vector, typename Vector::iterator>::type IteratorVector;
//    IteratorVector iterator_output(10);
//    thrust::lower_bound(vec.begin(), vec.end(), input.begin(), input.end(), iterator_output.begin());
//
//    ASSERT_EQUAL(iterator_output[0] - vec.begin(), 0);
//    ASSERT_EQUAL(iterator_output[1] - vec.begin(), 1);
//    ASSERT_EQUAL(iterator_output[2] - vec.begin(), 1);
//    ASSERT_EQUAL(iterator_output[3] - vec.begin(), 2);
//    ASSERT_EQUAL(iterator_output[4] - vec.begin(), 2);
//    ASSERT_EQUAL(iterator_output[5] - vec.begin(), 2);
//    ASSERT_EQUAL(iterator_output[6] - vec.begin(), 3);
//    ASSERT_EQUAL(iterator_output[7] - vec.begin(), 3);
//    ASSERT_EQUAL(iterator_output[8] - vec.begin(), 4);
//    ASSERT_EQUAL(iterator_output[9] - vec.begin(), 5);
}
DECLARE_VECTOR_UNITTEST(TestVectorLowerBoundSimple);


template <class Vector>
void TestVectorUpperBoundSimple(void)
{
    typedef typename Vector::value_type T;

    Vector vec(5);

    vec[0] = 0;
    vec[1] = 2;
    vec[2] = 5;
    vec[3] = 7;
    vec[4] = 8;

    Vector input(10);
    thrust::sequence(input.begin(), input.end());

    typedef typename vector_like<Vector, int>::type IntVector;

    // test with integral output type
    IntVector integral_output(10);
    typename IntVector::iterator output_end = thrust::experimental::upper_bound(vec.begin(), vec.end(), input.begin(), input.end(), integral_output.begin());

    ASSERT_EQUAL((output_end - integral_output.begin()), 10);

    ASSERT_EQUAL(integral_output[0], 1);
    ASSERT_EQUAL(integral_output[1], 1);
    ASSERT_EQUAL(integral_output[2], 2);
    ASSERT_EQUAL(integral_output[3], 2);
    ASSERT_EQUAL(integral_output[4], 2);
    ASSERT_EQUAL(integral_output[5], 3);
    ASSERT_EQUAL(integral_output[6], 3);
    ASSERT_EQUAL(integral_output[7], 4);
    ASSERT_EQUAL(integral_output[8], 5);
    ASSERT_EQUAL(integral_output[9], 5);

//    // test with interator output type
//    typedef typename vector_like<Vector, typename Vector::iterator>::type IteratorVector;
//    IteratorVector iterator_output(10);
//    thrust::lower_bound(vec.begin(), vec.end(), input.begin(), input.end(), iterator_output.begin());
//
//    ASSERT_EQUAL(iterator_output[0] - vec.begin(), 1);
//    ASSERT_EQUAL(iterator_output[1] - vec.begin(), 1);
//    ASSERT_EQUAL(iterator_output[2] - vec.begin(), 2);
//    ASSERT_EQUAL(iterator_output[3] - vec.begin(), 2);
//    ASSERT_EQUAL(iterator_output[4] - vec.begin(), 2);
//    ASSERT_EQUAL(iterator_output[5] - vec.begin(), 3);
//    ASSERT_EQUAL(iterator_output[6] - vec.begin(), 3);
//    ASSERT_EQUAL(iterator_output[7] - vec.begin(), 4);
//    ASSERT_EQUAL(iterator_output[8] - vec.begin(), 5);
//    ASSERT_EQUAL(iterator_output[9] - vec.begin(), 5);
}
DECLARE_VECTOR_UNITTEST(TestVectorUpperBoundSimple);


template <class Vector>
void TestVectorBinarySearchSimple(void)
{
    typedef typename Vector::value_type T;

    Vector vec(5);

    vec[0] = 0;
    vec[1] = 2;
    vec[2] = 5;
    vec[3] = 7;
    vec[4] = 8;

    Vector input(10);
    thrust::sequence(input.begin(), input.end());

    typedef typename vector_like<Vector, bool>::type BoolVector;
    typedef typename vector_like<Vector,  int>::type IntVector;

    // test with boolean output type
    BoolVector bool_output(10);
    typename BoolVector::iterator bool_output_end = thrust::experimental::binary_search(vec.begin(), vec.end(), input.begin(), input.end(), bool_output.begin());

    ASSERT_EQUAL((bool_output_end - bool_output.begin()), 10);

    ASSERT_EQUAL(bool_output[0],  true);
    ASSERT_EQUAL(bool_output[1], false);
    ASSERT_EQUAL(bool_output[2],  true);
    ASSERT_EQUAL(bool_output[3], false);
    ASSERT_EQUAL(bool_output[4], false);
    ASSERT_EQUAL(bool_output[5],  true);
    ASSERT_EQUAL(bool_output[6], false);
    ASSERT_EQUAL(bool_output[7],  true);
    ASSERT_EQUAL(bool_output[8],  true);
    ASSERT_EQUAL(bool_output[9], false);
    
    // test with integral output type
    IntVector integral_output(10, 2);
    typename IntVector::iterator int_output_end = thrust::experimental::binary_search(vec.begin(), vec.end(), input.begin(), input.end(), integral_output.begin());

    ASSERT_EQUAL((int_output_end - integral_output.begin()), 10);
    
    ASSERT_EQUAL(integral_output[0], 1);
    ASSERT_EQUAL(integral_output[1], 0);
    ASSERT_EQUAL(integral_output[2], 1);
    ASSERT_EQUAL(integral_output[3], 0);
    ASSERT_EQUAL(integral_output[4], 0);
    ASSERT_EQUAL(integral_output[5], 1);
    ASSERT_EQUAL(integral_output[6], 0);
    ASSERT_EQUAL(integral_output[7], 1);
    ASSERT_EQUAL(integral_output[8], 1);
    ASSERT_EQUAL(integral_output[9], 0);
}
DECLARE_VECTOR_UNITTEST(TestVectorBinarySearchSimple);


template <typename T>
struct TestVectorLowerBound
{
  void operator()(const size_t n)
  {
    thrust::host_vector<T>   h_vec = thrusttest::random_integers<T>(n);
    thrust::device_vector<T> d_vec = h_vec;

    thrust::sort(h_vec.begin(), h_vec.end());
    thrust::sort(d_vec.begin(), d_vec.end());

    thrust::host_vector<T>   h_input = thrusttest::random_integers<T>(4*n);
    thrust::device_vector<T> d_input = h_input;
    
    thrust::host_vector<int>   h_output(4*n);
    thrust::device_vector<int> d_output(4*n);

    thrust::experimental::lower_bound(h_vec.begin(), h_vec.end(), h_input.begin(), h_input.end(), h_output.begin());
    thrust::experimental::lower_bound(d_vec.begin(), d_vec.end(), d_input.begin(), d_input.end(), d_output.begin());

    ASSERT_ALMOST_EQUAL(h_output, d_output);
  }
};
VariableUnitTest<TestVectorLowerBound, NumericTypes> TestVectorLowerBoundInstance;


template <typename T>
struct TestVectorUpperBound
{
  void operator()(const size_t n)
  {
    thrust::host_vector<T>   h_vec = thrusttest::random_integers<T>(n);
    thrust::device_vector<T> d_vec = h_vec;

    thrust::sort(h_vec.begin(), h_vec.end());
    thrust::sort(d_vec.begin(), d_vec.end());

    thrust::host_vector<T>   h_input = thrusttest::random_integers<T>(4*n);
    thrust::device_vector<T> d_input = h_input;
    
    thrust::host_vector<int>   h_output(4*n);
    thrust::device_vector<int> d_output(4*n);

    thrust::experimental::upper_bound(h_vec.begin(), h_vec.end(), h_input.begin(), h_input.end(), h_output.begin());
    thrust::experimental::upper_bound(d_vec.begin(), d_vec.end(), d_input.begin(), d_input.end(), d_output.begin());

    ASSERT_ALMOST_EQUAL(h_output, d_output);
  }
};
VariableUnitTest<TestVectorUpperBound, NumericTypes> TestVectorUpperBoundInstance;

template <typename T>
struct TestVectorBinarySearch
{
  void operator()(const size_t n)
  {
    thrust::host_vector<T>   h_vec = thrusttest::random_integers<T>(n);
    thrust::device_vector<T> d_vec = h_vec;

    thrust::sort(h_vec.begin(), h_vec.end());
    thrust::sort(d_vec.begin(), d_vec.end());

    thrust::host_vector<T>   h_input = thrusttest::random_integers<T>(4*n);
    thrust::device_vector<T> d_input = h_input;
    
    thrust::host_vector<int>   h_output(4*n);
    thrust::device_vector<int> d_output(4*n);

    thrust::experimental::binary_search(h_vec.begin(), h_vec.end(), h_input.begin(), h_input.end(), h_output.begin());
    thrust::experimental::binary_search(d_vec.begin(), d_vec.end(), d_input.begin(), d_input.end(), d_output.begin());

    ASSERT_ALMOST_EQUAL(h_output, d_output);
  }
};
VariableUnitTest<TestVectorBinarySearch, NumericTypes> TestVectorBinarySearchInstance;

