#include <unittest/unittest.h>
#include <thrust/set_operations.h>
#include <thrust/functional.h>
#include <thrust/sort.h>
#include <thrust/iterator/discard_iterator.h>


template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator>
OutputIterator set_intersection(my_system &system,
                                InputIterator1,
                                InputIterator1,
                                InputIterator2,
                                InputIterator2,
                                OutputIterator result)
{
  system.validate_dispatch();
  return result;
}

void TestSetIntersectionDispatchExplicit()
{
  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::set_intersection(sys,
                           vec.begin(),
                           vec.begin(),
                           vec.begin(),
                           vec.begin(),
                           vec.begin());

  ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestSetIntersectionDispatchExplicit);


template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator>
OutputIterator set_intersection(my_tag,
                                InputIterator1,
                                InputIterator1,
                                InputIterator2,
                                InputIterator2,
                                OutputIterator result)
{
  *result = 13;
  return result;
}

void TestSetIntersectionDispatchImplicit()
{
  thrust::device_vector<int> vec(1);

  thrust::set_intersection(thrust::retag<my_tag>(vec.begin()),
                           thrust::retag<my_tag>(vec.begin()),
                           thrust::retag<my_tag>(vec.begin()),
                           thrust::retag<my_tag>(vec.begin()),
                           thrust::retag<my_tag>(vec.begin()));

  ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestSetIntersectionDispatchImplicit);


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
  size_t sizes[]   = {0, 1, n / 2, n, n + 1, 2 * n};
  size_t num_sizes = sizeof(sizes) / sizeof(size_t);

  thrust::host_vector<T> random = unittest::random_integers<unittest::int8_t>(n + *thrust::max_element(sizes, sizes + num_sizes));

  thrust::host_vector<T> h_a(random.begin(), random.begin() + n);
  thrust::host_vector<T> h_b(random.begin() + n, random.end());

  thrust::stable_sort(h_a.begin(), h_a.end());
  thrust::stable_sort(h_b.begin(), h_b.end());
  
  thrust::device_vector<T> d_a = h_a;
  thrust::device_vector<T> d_b = h_b;

  for (size_t i = 0; i < num_sizes; i++)
  {
    size_t size = sizes[i];
    
    thrust::host_vector<T>   h_result(n + size);
    thrust::device_vector<T> d_result(n + size);

    typename thrust::host_vector<T>::iterator   h_end;
    typename thrust::device_vector<T>::iterator d_end;
    
    h_end = thrust::set_intersection(h_a.begin(), h_a.end(),
                                     h_b.begin(), h_b.begin() + size,
                                     h_result.begin());
    h_result.resize(h_end - h_result.begin());

    d_end = thrust::set_intersection(d_a.begin(), d_a.end(),
                                     d_b.begin(), d_b.begin() + size,
                                     d_result.begin());
    d_result.resize(d_end - d_result.begin());

    ASSERT_EQUAL(h_result, d_result);
  }
}
DECLARE_VARIABLE_UNITTEST(TestSetIntersection);


template<typename T>
void TestSetIntersectionToDiscardIterator(const size_t n)
{
  thrust::host_vector<T> temp = unittest::random_integers<T>(2 * n);
  thrust::host_vector<T> h_a(temp.begin(), temp.begin() + n);
  thrust::host_vector<T> h_b(temp.begin() + n, temp.end());

  thrust::sort(h_a.begin(), h_a.end());
  thrust::sort(h_b.begin(), h_b.end());

  thrust::device_vector<T> d_a = h_a;
  thrust::device_vector<T> d_b = h_b;

  thrust::discard_iterator<> h_result;
  thrust::discard_iterator<> d_result;

  thrust::host_vector<T> h_reference(n);
  typename thrust::host_vector<T>::iterator h_end = 
    thrust::set_intersection(h_a.begin(), h_a.end(),
                             h_b.begin(), h_b.end(),
                             h_reference.begin());
  h_reference.erase(h_end, h_reference.end());
  
  h_result = thrust::set_intersection(h_a.begin(), h_a.end(),
                                      h_b.begin(), h_b.end(),
                                      thrust::make_discard_iterator());

  d_result = thrust::set_intersection(d_a.begin(), d_a.end(),
                                      d_b.begin(), d_b.end(),
                                      thrust::make_discard_iterator());

  thrust::discard_iterator<> reference(h_reference.size());

  ASSERT_EQUAL_QUIET(reference, h_result);
  ASSERT_EQUAL_QUIET(reference, d_result);
}
DECLARE_VARIABLE_UNITTEST(TestSetIntersectionToDiscardIterator);


template<typename T>
void TestSetIntersectionEquivalentRanges(const size_t n)
{
  thrust::host_vector<T> temp = unittest::random_integers<T>(n);
  thrust::host_vector<T> h_a = temp; thrust::sort(h_a.begin(), h_a.end());
  thrust::host_vector<T> h_b = h_a;

  thrust::device_vector<T> d_a = h_a;
  thrust::device_vector<T> d_b = h_b;

  thrust::host_vector<T>   h_result(n);
  thrust::device_vector<T> d_result(n);

  typename thrust::host_vector<T>::iterator   h_end;
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
}
DECLARE_VARIABLE_UNITTEST(TestSetIntersectionEquivalentRanges);


template<typename T>
void TestSetIntersectionMultiset(const size_t n)
{
  thrust::host_vector<T> temp = unittest::random_integers<T>(2 * n);

  // restrict elements to [min,13)
  for(typename thrust::host_vector<T>::iterator i = temp.begin();
      i != temp.end();
      ++i)
  {
    int temp = static_cast<int>(*i);
    temp %= 13;
    *i = temp;
  }

  thrust::host_vector<T> h_a(temp.begin(), temp.begin() + n);
  thrust::host_vector<T> h_b(temp.begin() + n, temp.end());

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
}
DECLARE_VARIABLE_UNITTEST(TestSetIntersectionMultiset);

