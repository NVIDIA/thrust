#include <thrust/count.h>

template <class Policy,
          typename Container,
          typename EqualityComparable = typename Container::value_type>
struct Find
{
  Container A;
  EqualityComparable value;
  Policy policy;

  template <typename Range>
  Find(Policy policy_, const Range& X, EqualityComparable value)
    : A(X.begin(), X.end()),
      value(value),
      policy(policy_)
  {}

  void operator()(void)
  {
    thrust::find(policy,A.begin(), A.end(), value);
  }
};

template <class Policy,
          typename Container,
          typename Predicate = thrust::identity<typename Container::value_type> >
struct FindIf
{
  Container A;
  Predicate pred;
  Policy policy;

  template <typename Range>
  FindIf(Policy policy_, const Range& X, Predicate pred = Predicate())
    : A(X.begin(), X.end()),
      pred(pred),
      policy(policy_)
  {}

  void operator()(void)
  {
    thrust::find_if(policy,A.begin(), A.end(), pred);
  }
};

template <class Policy,
          typename Container,
          typename Predicate = thrust::identity<typename Container::value_type> >
struct FindIfNot
{
  Container A;
  Predicate pred;
  Policy policy;

  template <typename Range>
  FindIfNot(Policy policy_, const Range& X, Predicate pred = Predicate())
    : A(X.begin(), X.end()),
      pred(pred),
      policy(policy_)
  {}

  void operator()(void)
  {
    thrust::find_if_not(policy,A.begin(), A.end(), pred);
  }
};

