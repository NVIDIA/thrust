#include <unittest/unittest.h>
#include <thrust/detail/device/generic/scalar/select.h>
#include <thrust/functional.h>
#include <thrust/sort.h>
#include <thrust/merge.h>

template<typename Iterator1, typename Iterator2>
  struct select_functor
{
  Iterator1 first1, last1;
  Iterator2 first2, last2;
  int k;

  select_functor(Iterator1 f1, Iterator1 l1,
                 Iterator2 f2, Iterator2 l2,
                 int kk)
    : first1(f1), last1(l1), first2(f2), last2(l2), k(kk)
  {}

  template<typename Dummy>
  __host__ __device__
  typename thrust::iterator_value<Iterator1>::type operator()(Dummy)
  {
    typedef typename thrust::iterator_value<Iterator1>::type value_type;
    return thrust::detail::device::generic::scalar::select(first1, last1, first2, last2, k, thrust::less<value_type>());
  }
};

template<typename T>
  void TestSelect(const size_t n)
{
  if(n == 0) return;

  typedef typename thrust::device_vector<T>::iterator iterator;

  thrust::host_vector<T> h_A = unittest::random_integers<T>(n);
  thrust::host_vector<T> h_B = unittest::random_integers<T>(n);

  // A and B must be sorted
  thrust::stable_sort(h_A.begin(), h_A.end());
  thrust::stable_sort(h_B.begin(), h_B.end());

  thrust::device_vector<T> d_A = h_A;
  thrust::device_vector<T> d_B = h_B;

  // merge A and B to create a reference
  thrust::host_vector<T> ref;
  ref.insert(ref.end(), h_A.begin(), h_A.end());
  ref.insert(ref.end(), h_B.begin(), h_B.end());
  thrust::merge(h_A.begin(), h_A.end(), h_B.begin(), h_B.end(), ref.begin());

  // choose some interesting values for k
  const size_t n_k = 6;
  const int k[n_k] = {0, n-1, n/2, n, thrust::min(n+1, 2*n-1), 2*n-1};

  for(size_t i = 0; i < n_k; ++i)
  {
    // test device
    thrust::device_vector<T> result(1);

    select_functor<iterator,iterator> f(d_A.begin(), d_A.end(),
                                        d_B.begin(), d_B.end(),
                                        k[i]);

    thrust::transform(thrust::make_counting_iterator(0u),
                      thrust::make_counting_iterator(1u),
                      result.begin(),
                      f);

    ASSERT_EQUAL(ref[k[i]], (T) result[0]);
  }
}
DECLARE_VARIABLE_UNITTEST(TestSelect);

template<typename U>
  void TestSelectKeyValue(const size_t n)
{
  if(n == 0) return;

  typedef key_value<U,U> T;

  typedef typename thrust::device_vector<T>::iterator iterator;

  thrust::host_vector<U> h_keys_A   = unittest::random_integers<U>(n);
  thrust::host_vector<U> h_values_A = unittest::random_integers<U>(n);

  thrust::host_vector<U> h_keys_B   = unittest::random_integers<U>(n);
  thrust::host_vector<U> h_values_B = unittest::random_integers<U>(n);

  thrust::host_vector<T> h_A(n), h_B(n);
  for(size_t i = 0; i < n; ++i)
  {
    h_A[i] = T(h_keys_A[i], h_values_A[i]);
    h_B[i] = T(h_keys_B[i], h_values_B[i]);
  }

  // A and B must be sorted
  thrust::stable_sort(h_A.begin(), h_A.end());
  thrust::stable_sort(h_B.begin(), h_B.end());

  thrust::device_vector<T> d_A = h_A;
  thrust::device_vector<T> d_B = h_B;

  // merge A and B to create a reference
  thrust::host_vector<T> ref;
  ref.insert(ref.end(), h_A.begin(), h_A.end());
  ref.insert(ref.end(), h_B.begin(), h_B.end());
  thrust::merge(h_A.begin(), h_A.end(), h_B.begin(), h_B.end(), ref.begin());

  // choose some interesting values for k
  const size_t n_k = 6;
  const int k[n_k] = {0, n-1, n/2, n, thrust::min(n+1, 2*n-1), 2*n-1};

  for(size_t i = 0; i < n_k; ++i)
  {
    // test device
    thrust::device_vector<T> result(1);

    select_functor<iterator,iterator> f(d_A.begin(), d_A.end(),
                                        d_B.begin(), d_B.end(),
                                        k[i]);

    thrust::transform(thrust::make_counting_iterator(0u),
                      thrust::make_counting_iterator(1u),
                      result.begin(),
                      f);

    ASSERT_EQUAL(ref[k[i]], (T) result[0]);
  }
}
DECLARE_VARIABLE_UNITTEST(TestSelectKeyValue);

