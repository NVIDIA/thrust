#include <unittest/unittest.h>
#include <thrust/swap.h>

#include <thrust/iterator/detail/forced_iterator.h> 

template <class Vector>
void TestSwapRangesSimple(void)
{
    typedef typename Vector::value_type T;

    Vector v1(5);
    v1[0] = 0; v1[1] = 1; v1[2] = 2; v1[3] = 3; v1[4] = 4;

    Vector v2(5);
    v2[0] = 5; v2[1] = 6; v2[2] = 7; v2[3] = 8; v2[4] = 9;

    thrust::swap_ranges(v1.begin(), v1.end(), v2.begin());

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
DECLARE_VECTOR_UNITTEST(TestSwapRangesSimple);


template <typename T>
void TestSwapRanges(const size_t n)
{
    thrust::host_vector<T> a1 = unittest::random_integers<T>(n);
    thrust::host_vector<T> a2 = unittest::random_integers<T>(n);

    thrust::host_vector<T>    h1 = a1;
    thrust::host_vector<T>    h2 = a2;
    thrust::device_vector<T>  d1 = a1;
    thrust::device_vector<T>  d2 = a2;
  
    thrust::swap_ranges(h1.begin(), h1.end(), h2.begin());
    thrust::swap_ranges(d1.begin(), d1.end(), d2.begin());

    ASSERT_EQUAL(h1, a2);  
    ASSERT_EQUAL(d1, a2);
    ASSERT_EQUAL(h2, a1);
    ASSERT_EQUAL(d2, a1);
}
DECLARE_VARIABLE_UNITTEST(TestSwapRanges);

#if (THRUST_DEVICE_BACKEND == THRUST_DEVICE_BACKEND_OMP)
void TestSwapRangesForcedIterator(void)
{
  thrust::device_vector<int> A(3, 0);
  thrust::device_vector<int> B(3, 1);

  thrust::swap_ranges(thrust::detail::make_forced_iterator(A.begin(), thrust::host_space_tag()),
                      thrust::detail::make_forced_iterator(A.end(),   thrust::host_space_tag()),
                      thrust::detail::make_forced_iterator(B.begin(), thrust::host_space_tag()));

  ASSERT_EQUAL(A[0], 1);
  ASSERT_EQUAL(A[1], 1);
  ASSERT_EQUAL(A[2], 1);
  ASSERT_EQUAL(B[0], 0);
  ASSERT_EQUAL(B[1], 0);
  ASSERT_EQUAL(B[2], 0);
}
DECLARE_UNITTEST(TestSwapRangesForcedIterator);
#endif

