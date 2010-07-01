#include <thrust/transform.h>

// [2.5.3.4] Transform

// all tests return void
// all test function names begin with "Test"
// test function template parameter names specify which the types the factory should invoke with
// we only recognize a limited set of template parameter names
template<typename InputIterator, typename OutputIterator, typename UnaryOperation>
void
TestTransformUnaryGenerality(InputIterator first,
                             InputIterator last,
                             OutputIterator result,
                             UnaryOperation op)
{
  // XXX this test won't actually work for general Input & Output Iterators
  // XXX we might have to dispatch on traversal inside the tests
  typedef typename thrust::iterator_value<InputIterator>::type  input_type;
  typedef typename thrust::iterator_value<OutputIterator>::type output_type;

  // ensure that the result of transform on the given ranges matches on the host & device
  thrust::host_vector<input_type>    h_input(first,last);
  thrust::host_vector<output_type>   h_output(h_input.size());

  thrust::device_vector<input_type>  d_input(first,last);
  thrust::device_vector<output_type> d_output(d_input.size());

  // run algo on input
  OutputIterator result_of_transform = thrust::transform(first, last, result, op);

  // check result_of_transform is result + size
  ASSERT_EQUAL(thrust::advance(result,h_input.size()), result_of_transform,
               "Incorrect iterator return value.");

  // run algo on host
  thrust::transform(h_input.begin(), h_input.end(), h_output.begin(), op);

  // run algo on device
  thrust::transform(d_input.begin(), d_input.end(), d_output.begin(), op);

  // check input result against host
  thrust::host_vector<output_type> h_result_temp(result, thrust::advance(h_result.size()));
  ASSERT_EQUAL(h_output, h_result_temp,
               "Test result doesn't match host's result.");

  // check input result against device
  thrust::device_vector<output_type> d_result_temp(result, thrust::advance(d_result.size()));
  ASSERT_EQUAL(d_output, d_result_temp,
               "Test result doesn't match device's result.");
}

template<typename Space>
void
TestTransformUnarySemantics(void)
{
  typedef int T;

  thrusttest::vector<T,Space> input(3);
  thrusttest::vector<T,Space> output(3);

  thrusttest::vector<T,Space> reference(3);

  input[0]     =  1; input[1]     = -2; input[2]     =  3;
  reference[0] = -1; reference[1] =  2; reference[2] = -3;

  typename thrusttest::vector<T,Space>::iterator iter =
    thrust::transform(input.begin(), input.end(), output.begin(), thrust::negate<T>());
  
  // assert we returned the end of the output
  ASSERT_EQUAL_QUIET(iter, output.end(),
                     "Incorrect iterator return value.");

  // assert the output matches the reference
  ASSERT_EQUAL(reference, output,
               "Test result doesn't match reference.");
}
