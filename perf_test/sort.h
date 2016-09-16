#include <thrust/sort.h>

template <class Policy,
          typename Container,
          typename StrictWeakOrdering = thrust::less<typename Container::value_type> >
struct Sort
{
  Container A, A_copy;
  StrictWeakOrdering comp;
  Policy policy;

  template <typename Range>
  Sort(Policy p_, const Range& X, StrictWeakOrdering comp = StrictWeakOrdering())
    : A(X.begin(), X.end()), A_copy(X.begin(), X.end()),
      comp(comp),
      policy(p_)
  {}

  void operator()(void)
  {
    thrust::sort(policy, A.begin(), A.end(), comp);
  }

  void reset(void)
  {
    thrust::copy(policy, A_copy.begin(), A_copy.end(), A.begin());
  }
};

template <typename T>
struct MyCompare
  : private thrust::less<T>
{
  inline __host__ __device__
  bool operator()(const T& x, const T &y) const
  {
    return thrust::less<T>::operator()(x,y);
  }
};

template <class Policy, typename Container>
struct ComparisonSort
  : Sort<Policy, Container, MyCompare<typename Container::value_type> >
{
  typedef Sort<Policy, Container, MyCompare<typename Container::value_type> > super_t;

  template <typename Range>
  ComparisonSort(Policy p_, const Range& X)
    : super_t(p_, X)
  {}
};

template <class Policy,
          typename Container,
          typename StrictWeakOrdering = thrust::less<typename Container::value_type> >
struct StableSort
{
  Container A, A_copy;
  StrictWeakOrdering comp;
  Policy policy;

  template <typename Range>
  StableSort(Policy p_, const Range& X, StrictWeakOrdering comp = StrictWeakOrdering())
    : A(X.begin(), X.end()), A_copy(X.begin(), X.end()),
      comp(comp),
      policy(p_)
  {}

  void operator()(void)
  {
    thrust::stable_sort(policy, A.begin(), A.end(), comp);
  }

  void reset(void)
  {
    thrust::copy(policy, A_copy.begin(), A_copy.end(), A.begin());
  }
};

template <class Policy,
          typename Container1,
          typename Container2 = Container1,
          typename StrictWeakOrdering = thrust::less<typename Container1::value_type> >
struct SortByKey
{
  Container1 A, A_copy; // keys
  Container2 B, B_copy; // values
  StrictWeakOrdering comp;
  Policy policy;

  template <typename Range1, typename Range2>
  SortByKey(Policy p_, const Range1& X, const Range2& Y, StrictWeakOrdering comp = StrictWeakOrdering())
    : A(X.begin(), X.end()), A_copy(X.begin(), X.end()),
      B(Y.begin(), Y.end()), B_copy(Y.begin(), Y.end()),
      comp(comp),
      policy(p_)
  {}

  void operator()(void)
  {
    thrust::sort_by_key(A.begin(), A.end(), B.begin(), comp);
  }

  void reset(void)
  {
    thrust::copy(policy, A_copy.begin(), A_copy.end(), A.begin());
    thrust::copy(policy, B_copy.begin(), B_copy.end(), B.begin());
  }
};


template <class Policy,
          typename Container1,
          typename Container2 = Container1>
struct ComparisonSortByKey
  : SortByKey<Policy, Container1, Container2, MyCompare<typename Container1::value_type> >
{
  typedef SortByKey<Policy, Container1, Container2, MyCompare<typename Container1::value_type> > super_t;

  template <typename Range1, typename Range2>
  ComparisonSortByKey(Policy p_, const Range1& X, const Range2& Y)
    : super_t(p_, X,Y)
  {}
};

template <class Policy,
          typename Container1,
          typename Container2 = Container1,
          typename StrictWeakOrdering = thrust::less<typename Container1::value_type> >
struct StableSortByKey
{
  Container1 A, A_copy; // keys
  Container2 B, B_copy; // values
  StrictWeakOrdering comp;
  Policy policy;

  template <typename Range1, typename Range2>
  StableSortByKey(Policy p_, const Range1& X, const Range2& Y, StrictWeakOrdering comp = StrictWeakOrdering())
    : A(X.begin(), X.end()), A_copy(X.begin(), X.end()),
      B(Y.begin(), Y.end()), B_copy(Y.begin(), Y.end()),
      comp(comp),
      policy(p_)
  {}

  void operator()(void)
  {
    thrust::stable_sort_by_key(policy, A.begin(), A.end(), B.begin(), comp);
  }

  void reset(void)
  {
    thrust::copy(policy, A_copy.begin(), A_copy.end(), A.begin());
    thrust::copy(policy, B_copy.begin(), B_copy.end(), B.begin());
  }
};


template <class Policy,
          typename Container,
          typename StrictWeakOrdering = thrust::less<typename Container::value_type> >
struct IsSorted
{
  Container A;
  StrictWeakOrdering comp;
  Policy policy;

  template <typename Range>
  IsSorted(Policy p_, const Range& X, StrictWeakOrdering comp = StrictWeakOrdering())
    : A(X.begin(), X.end()),
      comp(comp),
      policy(p_)
  {}

  void operator()(void)
  {
    thrust::is_sorted(policy, A.begin(), A.end(), comp);
  }
};

template <class Policy,
          typename Container,
          typename StrictWeakOrdering = thrust::less<typename Container::value_type> >
struct IsSortedUntil
{
  Container A;
  StrictWeakOrdering comp;
  Policy policy;

  template <typename Range>
  IsSortedUntil(Policy p_, const Range& X, StrictWeakOrdering comp = StrictWeakOrdering())
    : A(X.begin(), X.end()),
      comp(comp),
      policy(p_)
  {}

  void operator()(void)
  {
    thrust::is_sorted_until(policy, A.begin(), A.end(), comp);
  }
};

