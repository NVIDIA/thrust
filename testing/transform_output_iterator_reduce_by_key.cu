#include <unittest/unittest.h>

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>


template <typename T>
struct TestTransformOutputIteratorReduceByKey
{
  void operator()(const size_t n)
  {
    thrust::host_vector<T> h_keys = unittest::random_samples<T>(n);
    thrust::sort(h_keys.begin(), h_keys.end());
    thrust::device_vector<T> d_keys = h_keys;

    thrust::host_vector<T> h_values   = unittest::random_samples<T>(n);
    thrust::device_vector<T> d_values = h_values;

    thrust::host_vector<T> h_result(n);
    thrust::device_vector<T> d_result(n);

    // run on host
    thrust::reduce_by_key(thrust::host,
                          h_keys.begin(),
                          h_keys.end(),
                          thrust::make_transform_iterator(h_values.begin(), thrust::negate<T>()),
                          thrust::discard_iterator<T>{},
                          h_result.begin());
    // run on device
    thrust::reduce_by_key(thrust::device,
                          d_keys.begin(),
                          d_keys.end(),
                          d_values.begin(),
                          thrust::discard_iterator<T>{},
                          thrust::make_transform_output_iterator(d_result.begin(),
                                                                 thrust::negate<T>()));

    ASSERT_EQUAL(h_result, d_result);
  }
};
VariableUnitTest<TestTransformOutputIteratorReduceByKey, SignedIntegralTypes>
  TestTransformOutputIteratorReduceByKeyInstance;

