#include <unittest/unittest.h>
#include <thrust/tabulate.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>


template<typename Iterator, typename Function>
__global__
void tabulate_kernel(Iterator first, Iterator last, Function f)
{
  thrust::tabulate(thrust::seq, first, last, f);
}


void TestTabulateDeviceSeq()
{
  typedef thrust::device_vector<int> Vector;
  using namespace thrust::placeholders;
  typedef typename Vector::value_type T;
  
  Vector v(5);

  tabulate_kernel<<<1,1>>>(v.begin(), v.end(), thrust::identity<T>());

  ASSERT_EQUAL(v[0], 0);
  ASSERT_EQUAL(v[1], 1);
  ASSERT_EQUAL(v[2], 2);
  ASSERT_EQUAL(v[3], 3);
  ASSERT_EQUAL(v[4], 4);

  tabulate_kernel<<<1,1>>>(v.begin(), v.end(), -_1);

  ASSERT_EQUAL(v[0],  0);
  ASSERT_EQUAL(v[1], -1);
  ASSERT_EQUAL(v[2], -2);
  ASSERT_EQUAL(v[3], -3);
  ASSERT_EQUAL(v[4], -4);
  
  tabulate_kernel<<<1,1>>>(v.begin(), v.end(), _1 * _1 * _1);

  ASSERT_EQUAL(v[0], 0);
  ASSERT_EQUAL(v[1], 1);
  ASSERT_EQUAL(v[2], 8);
  ASSERT_EQUAL(v[3], 27);
  ASSERT_EQUAL(v[4], 64);
}
DECLARE_UNITTEST(TestTabulateDeviceSeq);

