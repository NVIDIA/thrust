#include <thrust/count.h>

template <class Policy,
          typename Container,
          typename EqualityComparable = typename Container::value_type>
struct Count
{
  Container A;
  EqualityComparable value;
  Policy policy;

  template <typename Range>
  Count(Policy policy_, const Range& X, EqualityComparable value = EqualityComparable())
    : A(X.begin(), X.end()),
      value(value), policy(policy_)
  {}

  void operator()(void)
  {
    thrust::count(policy, A.begin(), A.end(), value);
  }
};

template <class Policy,
          typename Container,
          typename Predicate = thrust::identity<typename Container::value_type> >
struct CountIf
{
  Container A;
  Predicate pred;
  Policy policy;

  template <typename Range>
  CountIf(Policy policy_, const Range& X, Predicate pred = Predicate())
    : A(X.begin(), X.end()),
      pred(pred), policy(policy_)
  {}

  void operator()(void)
  {
    thrust::count_if(policy, A.begin(), A.end(), pred);
  }
};

