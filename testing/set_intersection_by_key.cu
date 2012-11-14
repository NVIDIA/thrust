#include <unittest/unittest.h>
#include <thrust/set_operations.h>
#include <thrust/functional.h>
#include <thrust/sort.h>


template<typename InputIterator1,
         typename InputIterator2,
         typename InputIterator3,
         typename OutputIterator1,
         typename OutputIterator2>
thrust::pair<OutputIterator1,OutputIterator2>
  set_intersection_by_key(my_system &system,
                          InputIterator1,
                          InputIterator1,
                          InputIterator2,
                          InputIterator2,
                          InputIterator3,
                          OutputIterator1 keys_result,
                          OutputIterator2 values_result)
{
  system.validate_dispatch();
  return thrust::make_pair(keys_result, values_result);
}

void TestSetIntersectionByKeyDispatchExplicit()
{
  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::set_intersection_by_key(sys,
                                  vec.begin(),
                                  vec.begin(),
                                  vec.begin(),
                                  vec.begin(),
                                  vec.begin(),
                                  vec.begin(),
                                  vec.begin());

  ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestSetIntersectionByKeyDispatchExplicit);


template<typename InputIterator1,
         typename InputIterator2,
         typename InputIterator3,
         typename OutputIterator1,
         typename OutputIterator2>
thrust::pair<OutputIterator1,OutputIterator2>
  set_intersection_by_key(my_tag,
                          InputIterator1,
                          InputIterator1,
                          InputIterator2,
                          InputIterator2,
                          InputIterator3,
                          OutputIterator1 keys_result,
                          OutputIterator2 values_result)
{
  *keys_result = 13;
  return thrust::make_pair(keys_result,values_result);
}

void TestSetIntersectionByKeyDispatchImplicit()
{
  thrust::device_vector<int> vec(1);

  thrust::set_intersection_by_key(thrust::retag<my_tag>(vec.begin()),
                                  thrust::retag<my_tag>(vec.begin()),
                                  thrust::retag<my_tag>(vec.begin()),
                                  thrust::retag<my_tag>(vec.begin()),
                                  thrust::retag<my_tag>(vec.begin()),
                                  thrust::retag<my_tag>(vec.begin()),
                                  thrust::retag<my_tag>(vec.begin()));

  ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestSetIntersectionByKeyDispatchImplicit);


template<typename Vector>
void TestSetIntersectionByKeySimple(void)
{
  typedef typename Vector::iterator Iterator;

  Vector a_key(3), b_key(4);
  Vector a_val(3);

  a_key[0] = 0; a_key[1] = 2; a_key[2] = 4;
  a_val[0] = 0; a_val[1] = 0; a_val[2] = 0;

  b_key[0] = 0; b_key[1] = 3; b_key[2] = 3; b_key[3] = 4;

  Vector ref_key(2), ref_val(2);
  ref_key[0] = 0; ref_key[1] = 4;
  ref_val[0] = 0; ref_val[1] = 0;

  Vector result_key(2), result_val(2);

  thrust::pair<Iterator,Iterator> end =
    thrust::set_intersection_by_key(a_key.begin(), a_key.end(),
                                    b_key.begin(), b_key.end(),
                                    a_val.begin(),
                                    result_key.begin(),
                                    result_val.begin());

  ASSERT_EQUAL_QUIET(result_key.end(), end.first);
  ASSERT_EQUAL_QUIET(result_val.end(), end.second);
  ASSERT_EQUAL(ref_key, result_key);
  ASSERT_EQUAL(ref_val, result_val);
}
DECLARE_VECTOR_UNITTEST(TestSetIntersectionByKeySimple);


template<typename T>
void TestSetIntersectionByKey(const size_t n)
{
  thrust::host_vector<T> random_keys = unittest::random_integers<unittest::int8_t>(n);
  thrust::host_vector<T> random_vals = unittest::random_integers<unittest::int8_t>(n);

  size_t denominators[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  size_t num_denominators = sizeof(denominators) / sizeof(size_t);

  for(size_t i = 0; i < num_denominators; ++i)
  {
    size_t size_a = n / denominators[i];

    thrust::host_vector<T> h_a_keys(random_keys.begin(), random_keys.begin() + size_a);
    thrust::host_vector<T> h_b_keys(random_keys.begin() + size_a, random_keys.end());

    thrust::host_vector<T> h_a_vals(random_vals.begin(), random_vals.begin() + size_a);

    thrust::stable_sort(h_a_keys.begin(), h_a_keys.end());
    thrust::stable_sort(h_b_keys.begin(), h_b_keys.end());

    thrust::device_vector<T> d_a_keys = h_a_keys;
    thrust::device_vector<T> d_b_keys = h_b_keys;

    thrust::device_vector<T> d_a_vals = h_a_vals;

    thrust::host_vector<T> h_result_keys(n);
    thrust::host_vector<T> h_result_vals(n);

    thrust::device_vector<T> d_result_keys(n);
    thrust::device_vector<T> d_result_vals(n);


    thrust::pair<
      typename thrust::host_vector<T>::iterator,
      typename thrust::host_vector<T>::iterator
    > h_end;

    thrust::pair<
      typename thrust::device_vector<T>::iterator,
      typename thrust::device_vector<T>::iterator
    > d_end;


    h_end = thrust::set_intersection_by_key(h_a_keys.begin(), h_a_keys.end(),
                                            h_b_keys.begin(), h_b_keys.end(),
                                            h_a_vals.begin(),
                                            h_result_keys.begin(),
                                            h_result_vals.begin());
    h_result_keys.erase(h_end.first, h_result_keys.end());
    h_result_vals.erase(h_end.second, h_result_vals.end());

    d_end = thrust::set_intersection_by_key(d_a_keys.begin(), d_a_keys.end(),
                                            d_b_keys.begin(), d_b_keys.end(),
                                            d_a_vals.begin(),
                                            d_result_keys.begin(),
                                            d_result_vals.begin());
    d_result_keys.erase(d_end.first, d_result_keys.end());
    d_result_vals.erase(d_end.second, d_result_vals.end());

    ASSERT_EQUAL(h_result_keys, d_result_keys);
    ASSERT_EQUAL(h_result_vals, d_result_vals);
  }
}
DECLARE_VARIABLE_UNITTEST(TestSetIntersectionByKey);


template<typename T>
void TestSetIntersectionByKeyEquivalentRanges(const size_t n)
{
  thrust::host_vector<T> temp = unittest::random_integers<T>(n);

  thrust::host_vector<T> h_a_key = temp;
  thrust::sort(h_a_key.begin(), h_a_key.end());
  thrust::host_vector<T> h_b_key = h_a_key;

  thrust::host_vector<T> h_a_val = unittest::random_integers<T>(n);

  thrust::device_vector<T> d_a_key = h_a_key;
  thrust::device_vector<T> d_b_key = h_b_key;

  thrust::device_vector<T> d_a_val = h_a_val;

  thrust::host_vector<T>   h_result_key(n), h_result_val(n);
  thrust::device_vector<T> d_result_key(n), d_result_val(n);

  thrust::pair<
    typename thrust::host_vector<T>::iterator,
    typename thrust::host_vector<T>::iterator
  > h_end;
  
  thrust::pair<
    typename thrust::device_vector<T>::iterator,
    typename thrust::device_vector<T>::iterator
  > d_end;
  
  h_end = thrust::set_intersection_by_key(h_a_key.begin(), h_a_key.end(),
                                          h_b_key.begin(), h_b_key.end(),
                                          h_a_val.begin(),
                                          h_result_key.begin(),
                                          h_result_val.begin());
  h_result_key.erase(h_end.first,  h_result_key.end());
  h_result_val.erase(h_end.second, h_result_val.end());

  d_end = thrust::set_intersection_by_key(d_a_key.begin(), d_a_key.end(),
                                          d_b_key.begin(), d_b_key.end(),
                                          d_a_val.begin(),
                                          d_result_key.begin(),
                                          d_result_val.begin());
  d_result_key.erase(d_end.first,  d_result_key.end());
  d_result_val.erase(d_end.second, d_result_val.end());

  ASSERT_EQUAL(h_result_key, d_result_key);
  ASSERT_EQUAL(h_result_val, d_result_val);
}
DECLARE_VARIABLE_UNITTEST(TestSetIntersectionByKeyEquivalentRanges);


template<typename T>
void TestSetIntersectionByKeyMultiset(const size_t n)
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

  thrust::host_vector<T> h_a_key(temp.begin(), temp.begin() + n);
  thrust::host_vector<T> h_b_key(temp.begin() + n, temp.end());

  thrust::sort(h_a_key.begin(), h_a_key.end());
  thrust::sort(h_b_key.begin(), h_b_key.end());

  thrust::host_vector<T> h_a_val = unittest::random_integers<T>(n);

  thrust::device_vector<T> d_a_key = h_a_key;
  thrust::device_vector<T> d_b_key = h_b_key;

  thrust::device_vector<T> d_a_val = h_a_val;

  thrust::host_vector<T>   h_result_key(n), h_result_val(n);
  thrust::device_vector<T> d_result_key(n), d_result_val(n);

  thrust::pair<
    typename thrust::host_vector<T>::iterator,
    typename thrust::host_vector<T>::iterator
  > h_end;

  thrust::pair<
    typename thrust::device_vector<T>::iterator,
    typename thrust::device_vector<T>::iterator
  > d_end;
  
  h_end = thrust::set_intersection_by_key(h_a_key.begin(), h_a_key.end(),
                                          h_b_key.begin(), h_b_key.end(),
                                          h_a_val.begin(),
                                          h_result_key.begin(),
                                          h_result_val.begin());
  h_result_key.erase(h_end.first,  h_result_key.end());
  h_result_val.erase(h_end.second, h_result_val.end());

  d_end = thrust::set_intersection_by_key(d_a_key.begin(), d_a_key.end(),
                                          d_b_key.begin(), d_b_key.end(),
                                          d_a_val.begin(),
                                          d_result_key.begin(),
                                          d_result_val.begin());
  d_result_key.erase(d_end.first,  d_result_key.end());
  d_result_val.erase(d_end.second, d_result_val.end());

  ASSERT_EQUAL(h_result_key, d_result_key);
  ASSERT_EQUAL(h_result_val, d_result_val);
}
DECLARE_VARIABLE_UNITTEST(TestSetIntersectionByKeyMultiset);

