#include <thrust/sequence.h>

template <class Policy, typename Container>
struct Sequence
{
  Container A;
  Policy policy;

  template <typename Range>
  Sequence(Policy p_, const Range& X)
    : A(X.begin(), X.end()), policy(p_)
  {}

  void operator()(void)
  {
    thrust::sequence(policy, A.begin(), A.end());
  }
};

