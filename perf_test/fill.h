#include <thrust/fill.h>

template <class Policy,
          typename Container,
          typename T = typename Container::value_type>
struct Fill
{
  Container A;
  T value;
  Policy policy;

  template <typename Range>
  Fill(Policy policy_, const Range& X, T value = T())
    : A(X.begin(), X.end()),
      value(value), 
      policy(policy_)
  {}

  void operator()(void)
  {
    thrust::fill(policy, A.begin(), A.end(), value);
  }
};

template <class Policy,
          typename Container,
          typename T = typename Container::value_type>
struct FillN
{
  Container A;
  T value;
  Policy policy;

  template <typename Range>
  FillN(Policy policy_, const Range& X, T value = T())
    : A(X.begin(), X.end()),
      value(value), 
      policy(policy_)
  {}

  void operator()(void)
  {
    thrust::fill_n(policy, A.begin(), A.size(), value);
  }
};

