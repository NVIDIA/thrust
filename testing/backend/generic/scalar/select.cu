#include <unittest/unittest.h>
#include <thrust/detail/backend/generic/scalar/select.h>
#include <thrust/functional.h>
#include <thrust/sort.h>
#include <thrust/merge.h>
#include <thrust/sequence.h>

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
    return thrust::detail::backend::generic::scalar::select(first1, last1, first2, last2, k, thrust::less<value_type>());
  }
};

template<typename T>
  void TestSelect(const size_t n)
{
#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC && CUDA_VERSION < 3020
  KNOWN_FAILURE;
#else
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
#endif
}
DECLARE_VARIABLE_UNITTEST(TestSelect);

template<typename U>
  void TestSelectKeyValue(const size_t n)
{
#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC && CUDA_VERSION < 3020
  KNOWN_FAILURE;
#else
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
#endif
}
DECLARE_VARIABLE_UNITTEST(TestSelectKeyValue);


struct compare_first
{
  template <typename Tuple>
  __host__ __device__
  bool operator()(const Tuple& a, const Tuple& b)
  {
    return thrust::get<0>(a) < thrust::get<0>(b);
  }
};


void TestSelectSemantics(void)
{
  int A[] = {0,0,0,0,0,1,1,1,1,1,2,2,2,2,5,5,5,5,5,8,8,8,8};
  int X[] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
  int B[] = {0,0,1,1,2,2,2,2,4,4,4,4,4,5,5,5,5,5,6,6,7,7,8,8,8,9};
  int Y[] = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};

  // tests
  int K[][3] = { { 0,0,0}, 
                 { 4,0,0},
                 { 5,0,1},
                 { 6,0,1},
                 { 7,1,0},
                 { 8,1,0},
                 {11,1,0},
                 {11,1,0},
                 {12,1,1},
                 {13,1,1},
                 {14,2,0},
                 {17,2,0},
                 {18,2,1},
                 {21,2,1},
                 {22,4,1},
                 {26,4,1},
                 {27,5,0},
                 {31,5,0},
                 {32,5,1},
                 {36,5,1},
                 {37,6,1},
                 {39,7,1},
                 {41,8,0},
                 {45,8,1},
                 {48,9,1},
               };

  size_t An = sizeof(A) / sizeof(int);
  size_t Bn = sizeof(B) / sizeof(int);
  size_t N = sizeof(K) / sizeof(int[3]);
  
  using thrust::detail::backend::generic::scalar::select;

  for (size_t i = 0; i < N; i++)
  {
    thrust::tuple<int,int> result =
      select(thrust::make_zip_iterator(thrust::make_tuple(&A[0],&X[0])),
             thrust::make_zip_iterator(thrust::make_tuple(&A[0],&X[0])) + An,
             thrust::make_zip_iterator(thrust::make_tuple(&B[0],&Y[0])),
             thrust::make_zip_iterator(thrust::make_tuple(&B[0],&Y[0])) + Bn,
             K[i][0],
             compare_first());
    
    //std::cout << "Test #" << i << " " << K[i][0] << " " << K[i][1] << " " << K[i][2] << std::endl;
    //std::cout << "      " << i << " " << K[i][0] << " " << thrust::get<0>(result) << " " << thrust::get<1>(result) << std::endl;

    ASSERT_EQUAL(thrust::get<0>(result), K[i][1]);
    ASSERT_EQUAL(thrust::get<1>(result), K[i][2]);
  }
}
DECLARE_UNITTEST(TestSelectSemantics);

