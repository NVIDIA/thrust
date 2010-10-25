#include <unittest/unittest.h>
#include <thrust/reverse.h>

typedef unittest::type_list<char,short,int> ReverseTypes;

template<typename Vector>
void TestReverseSimple(void)
{
  Vector data(5);
  data[0] = 1;
  data[1] = 2;
  data[2] = 1;
  data[3] = 1;
  data[4] = 2;

  thrust::reverse(data.begin(), data.end());

  Vector ref(5);
  ref[0] = 2;
  ref[1] = 1;
  ref[2] = 1;
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
  input[2] = 1;
  input[3] = 1;
  input[4] = 2;

  Vector output(5);

  Iterator iter = thrust::reverse_copy(input.begin(), input.end(), output.begin());

  Vector ref(5);
  ref[0] = 2;
  ref[1] = 1;
  ref[2] = 1;
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

    thrust::reverse(h_data.begin(), h_data.end());
    thrust::reverse(d_data.begin(), d_data.end());

    ASSERT_EQUAL(h_data, d_data);
  }
};
VariableUnitTest<TestReverseCopy, ReverseTypes> TestReverseCopyInstance;

