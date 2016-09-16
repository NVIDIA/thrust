#include <thrust/logical.h>

template <class Policy,
          typename Container,
          typename Predicate = thrust::identity<typename Container::value_type> >
struct AllOf
{
  Container A;
  Predicate pred;
  Policy policy;

  template <typename Range>
  AllOf(Policy p_, const Range& X, Predicate pred = Predicate())
    : A(X.begin(), X.end()),
      pred(pred),
      policy(p_)
  {}

  void operator()(void)
  {
    thrust::all_of(policy, A.begin(), A.end(), pred);
  }
};

template <class Policy,
          typename Container,
          typename Predicate = thrust::identity<typename Container::value_type> >
struct AnyOf
{
  Container A;
  Predicate pred;
  Policy policy;

  template <typename Range>
  AnyOf(Policy p_, const Range& X, Predicate pred = Predicate())
    : A(X.begin(), X.end()),
      pred(pred),
      policy(p_)
  {}

  void operator()(void)
  {
    thrust::any_of(policy, A.begin(), A.end(), pred);
  }
};

template <class Policy,
          typename Container,
          typename Predicate = thrust::identity<typename Container::value_type> >
struct NoneOf
{
  Container A;
  Predicate pred;
  Policy policy;

  template <typename Range>
  NoneOf(Policy p_, const Range& X, Predicate pred = Predicate())
    : A(X.begin(), X.end()),
      pred(pred),
      policy(p_)
  {}

  void operator()(void)
  {
    thrust::none_of(policy, A.begin(), A.end(), pred);
  }
};


