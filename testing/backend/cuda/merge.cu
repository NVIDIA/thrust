#include <unittest/unittest.h>
#include <thrust/merge.h>
#include <thrust/functional.h>
#include <thrust/sort.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>


template<typename Iterator1, typename Iterator2, typename Iterator3, typename Iterator4>
__global__
void merge_kernel(Iterator1 first1, Iterator1 last1,
                  Iterator2 first2, Iterator2 last2,
                  Iterator3 result1,
                  Iterator4 result2)
{
  *result2 = thrust::merge(thrust::seq, first1, last1, first2, last2, result1);
}


void TestMergeDeviceSeq()
{
  thrust::device_vector<int> a(3), b(4);

  a[0] = 0; a[1] = 2; a[2] = 4;
  b[0] = 0; b[1] = 3; b[2] = 3; b[3] = 4;

  thrust::device_vector<int> ref(7);
  ref[0] = 0;
  ref[1] = 0;
  ref[2] = 2;
  ref[3] = 3;
  ref[4] = 3;
  ref[5] = 4;
  ref[6] = 4;

  thrust::device_vector<int> result(7);
  thrust::device_vector<thrust::device_vector<int>::iterator> result_end(1);

  merge_kernel<<<1,1>>>(a.begin(), a.end(),
                        b.begin(), b.end(),
                        result.begin(),
                        result_end.begin());
  thrust::device_vector<int>::iterator end = result_end[0];

  ASSERT_EQUAL_QUIET(result.end(), end);
  ASSERT_EQUAL(ref, result);
}
DECLARE_UNITTEST(TestMergeDeviceSeq);

