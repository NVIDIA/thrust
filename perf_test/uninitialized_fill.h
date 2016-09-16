#include <thrust/uninitialized_fill.h>

template <class Policy,
          typename Container,
          typename T = typename Container::value_type>
struct UninitializedFill
{
  Container A;
  T value;
  Policy policy;

  template <typename Range>
  UninitializedFill(Policy p_, const Range& X, T value = T())
    : A(X.begin(), X.end()),
      value(value),
      policy(p_)
  {}

  void operator()(void)
  {
    thrust::uninitialized_fill(policy, A.begin(), A.end(), value);
  }
};

template <class Policy,
          typename Container,
          typename T = typename Container::value_type>
struct UninitializedFillN
{
  Container A;
  T value;
  Policy policy;

  template <typename Range>
  UninitializedFillN(Policy p_, const Range& X, T value = T())
    : A(X.begin(), X.end()),
      value(value),
      policy(p_)
  {}

  void operator()(void)
  {
    thrust::uninitialized_fill_n(policy, A.begin(), A.size(), value);
  }
};

