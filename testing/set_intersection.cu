#include <thrusttest/unittest.h>
#include <thrust/set_intersection.h>
#include <thrust/functional.h>
#include <thrust/sort.h>

template<typename Vector>
void TestSetIntersectionSimple(void)
{
  typedef typename Vector::iterator Iterator;

  Vector a(3), b(4);

  a[0] = 0; a[1] = 2; a[2] = 4;
  b[0] = 0; b[1] = 3; b[2] = 3; b[3] = 4;

  Vector ref(2);
  ref[0] = 0; ref[1] = 4;

  Vector result(2);

  Iterator end = thrust::set_intersection(a.begin(), a.end(),
                                          b.begin(), b.end(),
                                          result.begin());

  ASSERT_EQUAL_QUIET(result.end(), end);
  ASSERT_EQUAL(ref, result);
}
DECLARE_VECTOR_UNITTEST(TestSetIntersectionSimple);


template<typename T>
void TestSetIntersection(const size_t n)
{
  KNOWN_FAILURE;

  thrust::host_vector<T> h_a = thrusttest::random_integers<T>(n);
  thrust::host_vector<T> h_b = thrusttest::random_integers<T>(n);

  thrust::sort(h_a.begin(), h_a.end());
  thrust::sort(h_b.begin(), h_b.end());

  thrust::device_vector<T> d_a = h_a;
  thrust::device_vector<T> d_b = h_b;

  thrust::host_vector<T> h_result(n);
  thrust::device_vector<T> d_result(n);

  typename thrust::host_vector<T>::iterator h_end;
  typename thrust::device_vector<T>::iterator d_end;
  
  h_end = thrust::set_intersection(h_a.begin(), h_a.end(),
                                   h_b.begin(), h_b.end(),
                                   h_result.begin());
  h_result.resize(h_end - h_result.begin());

  d_end = thrust::set_intersection(d_a.begin(), d_a.end(),
                                   d_b.begin(), d_b.end(),
                                   d_result.begin());
  d_result.resize(d_end - d_result.begin());

  ASSERT_EQUAL(h_result, d_result);
};
DECLARE_VARIABLE_UNITTEST(TestSetIntersection);


template<typename Vector>
void TestSetIntersectionAscendingSimple(void)
{
  typedef typename Vector::value_type T;
  typedef typename Vector::iterator Iterator;

  Vector a(3), b(4);

  a[0] = 4; a[1] = 2; a[2] = 0;
  b[0] = 4; b[1] = 3; b[2] = 3; b[3] = 0;

  Vector ref(2);
  ref[0] = 4; ref[1] = 0;

  Vector result(2);

  Iterator end = thrust::set_intersection(a.begin(), a.end(),
                                          b.begin(), b.end(),
                                          result.begin(),
                                          thrust::greater<T>());

  ASSERT_EQUAL_QUIET(result.end(), end);
  ASSERT_EQUAL(ref, result);
}
DECLARE_VECTOR_UNITTEST(TestSetIntersectionAscendingSimple);


