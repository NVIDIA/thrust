#include <thrust/reverse.h>

template <class Policy,
          typename Container,
          typename T = typename Container::value_type>
struct Reverse
{
  Container A, A_copy;
  Policy policy;

  template <typename Range>
  Reverse(Policy p_, const Range& X)
    : A(X.begin(), X.end()), A_copy(X.begin(), X.end()),
    policy(p_)
  {}

  void operator()(void)
  {
    thrust::reverse(policy, A.begin(), A.end());
  }
  
  void reset(void)
  {
    // restore initial data
    thrust::copy(policy, A_copy.begin(), A_copy.end(), A.begin());
  }
};

template <class Policy,
          typename Container1,
          typename Container2 = Container1>
struct ReverseCopy
{
  Container1 A;
  Container2 B;
  Policy policy;

  template <typename Range1, typename Range2>
  ReverseCopy(Policy p_, const Range1& X, const Range2& Y)
    : A(X.begin(), X.end()),
      B(Y.begin(), Y.end()),
      policy(p_)
  {}

  void operator()(void)
  {
    thrust::reverse_copy(policy, A.begin(), A.end(), B.begin());
  }
};

