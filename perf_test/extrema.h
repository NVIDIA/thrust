#include <thrust/extrema.h>

template <class Policy,
          typename Container,
          typename BinaryPredicate = thrust::less<typename Container::value_type> >
struct MinElement
{
  Container A;
  BinaryPredicate binary_pred;
  Policy policy;

  template <typename Range>
  MinElement(Policy policy_, const Range& X, BinaryPredicate binary_pred = BinaryPredicate())
    : A(X.begin(), X.end()),
      binary_pred(binary_pred), 
      policy(policy_)
  {}

  void operator()(void)
  {
    thrust::min_element(policy,A.begin(), A.end(), binary_pred);
  }
};


template <class Policy,
          typename Container,
          typename BinaryPredicate = thrust::less<typename Container::value_type> >
struct MaxElement
{
  Container A;
  BinaryPredicate binary_pred;
  Policy policy;

  template <typename Range>
  MaxElement(Policy policy_, const Range& X, BinaryPredicate binary_pred = BinaryPredicate())
    : A(X.begin(), X.end()),
      binary_pred(binary_pred),
      policy(policy_)
  {}

  void operator()(void)
  {
    thrust::max_element(policy,A.begin(), A.end(), binary_pred);
  }
};


template <class Policy,
          typename Container,
          typename BinaryPredicate = thrust::less<typename Container::value_type> >
struct MinMaxElement
{
  Container A;
  BinaryPredicate binary_pred;
  Policy policy;

  template <typename Range>
  MinMaxElement(Policy policy_, const Range& X, BinaryPredicate binary_pred = BinaryPredicate())
    : A(X.begin(), X.end()),
      binary_pred(binary_pred),
      policy(policy_)
  {}

  void operator()(void)
  {
    thrust::minmax_element(policy,A.begin(), A.end(), binary_pred);
  }
};

