#include <unittest/unittest.h>
#include <thrust/reverse.h>
#include <thrust/iterator/discard_iterator.h>

typedef unittest::type_list<char,short,int> ReverseTypes;

template<typename Vector>
void TestReverseSimple(void)
{
  Vector data(5);
  data[0] = 1;
  data[1] = 2;
  data[2] = 3;
  data[3] = 4;
  data[4] = 5;

  thrust::reverse(data.begin(), data.end());

  Vector ref(5);
  ref[0] = 5;
  ref[1] = 4;
  ref[2] = 3;
  ref[3] = 2;
  ref[4] = 1;

  ASSERT_EQUAL(ref, data);
}
DECLARE_VECTOR_UNITTEST(TestReverseSimple);

template<typename Vector>
void TestReverseCopySimple(void)
{
  typedef typename Vector::iterator   Iterator;

  Vector input(5);
  input[0] = 1;
  input[1] = 2;
  input[2] = 3;
  input[3] = 4;
  input[4] = 5;

  Vector output(5);

  Iterator iter = thrust::reverse_copy(input.begin(), input.end(), output.begin());

  Vector ref(5);
  ref[0] = 5;
  ref[1] = 4;
  ref[2] = 3;
  ref[3] = 2;
  ref[4] = 1;

  ASSERT_EQUAL(5, iter - output.begin());
  ASSERT_EQUAL(ref, output);
}
DECLARE_VECTOR_UNITTEST(TestReverseCopySimple);

template<typename T>
struct TestReverse
{
  void operator()(const size_t n)
  {
    thrust::host_vector<T> h_data = unittest::random_integers<T>(n);
    thrust::device_vector<T> d_data = h_data;

    thrust::reverse(h_data.begin(), h_data.end());
    thrust::reverse(d_data.begin(), d_data.end());

    ASSERT_EQUAL(h_data, d_data);
  }
};
VariableUnitTest<TestReverse, ReverseTypes> TestReverseInstance;

template<typename T>
struct TestReverseCopy
{
  void operator()(const size_t n)
  {
    thrust::host_vector<T> h_data = unittest::random_integers<T>(n);
    thrust::device_vector<T> d_data = h_data;

    thrust::host_vector<T> h_result(n);
    thrust::device_vector<T> d_result(n);

    thrust::reverse_copy(h_data.begin(), h_data.end(), h_result.begin());
    thrust::reverse_copy(d_data.begin(), d_data.end(), d_result.begin());

    ASSERT_EQUAL(h_result, d_result);
  }
};
VariableUnitTest<TestReverseCopy, ReverseTypes> TestReverseCopyInstance;

template<typename T>
struct TestReverseCopyToDiscardIterator
{
  void operator()(const size_t n)
  {
    thrust::host_vector<T> h_data = unittest::random_integers<T>(n);
    thrust::device_vector<T> d_data = h_data;

    thrust::discard_iterator<> h_result =
      thrust::reverse_copy(h_data.begin(), h_data.end(), thrust::make_discard_iterator());

    thrust::discard_iterator<> d_result =
      thrust::reverse_copy(d_data.begin(), d_data.end(), thrust::make_discard_iterator());

    thrust::discard_iterator<> reference(n);

    ASSERT_EQUAL_QUIET(reference, h_result);
    ASSERT_EQUAL_QUIET(reference, d_result);
  }
};
VariableUnitTest<TestReverseCopyToDiscardIterator, ReverseTypes> TestReverseCopyToDiscardIteratorInstance;

