#include <unittest/unittest.h>
#include <thrust/swap.h>
#include <thrust/execution_policy.h>


template<typename Iterator1, typename Iterator2>
__global__
void swap_ranges_kernel(Iterator1 first1, Iterator1 last1, Iterator2 first2)
{
  thrust::swap_ranges(thrust::seq, first1, last1, first2);
}


void TestSwapRangesDeviceSeq()
{
  typedef thrust::device_vector<int> Vector;
  typedef typename Vector::value_type T;

  Vector v1(5);
  v1[0] = 0; v1[1] = 1; v1[2] = 2; v1[3] = 3; v1[4] = 4;

  Vector v2(5);
  v2[0] = 5; v2[1] = 6; v2[2] = 7; v2[3] = 8; v2[4] = 9;

  swap_ranges_kernel<<<1,1>>>(v1.begin(), v1.end(), v2.begin());

  ASSERT_EQUAL(v1[0], 5);
  ASSERT_EQUAL(v1[1], 6);
  ASSERT_EQUAL(v1[2], 7);
  ASSERT_EQUAL(v1[3], 8);
  ASSERT_EQUAL(v1[4], 9);
  
  ASSERT_EQUAL(v2[0], 0);
  ASSERT_EQUAL(v2[1], 1);
  ASSERT_EQUAL(v2[2], 2);
  ASSERT_EQUAL(v2[3], 3);
  ASSERT_EQUAL(v2[4], 4);
}
DECLARE_UNITTEST(TestSwapRangesDeviceSeq);

