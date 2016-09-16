#include <thrust/swap.h>

template <class Policy,
          typename Container1,
          typename Container2 = Container1>
struct SwapRanges
{
  Container1 A;
  Container2 B;
  Policy policy;
 
  template <typename Range1, typename Range2>
  SwapRanges(Policy p_, const Range1& X, const Range2& Y)
    : A(X.begin(), X.end()),
      B(Y.begin(), Y.end()),
      policy(p_)
  {}

  void operator()(void)
  {
    thrust::swap_ranges(policy, A.begin(), A.end(), B.begin());
  }
};

