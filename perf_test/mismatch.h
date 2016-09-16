#include <thrust/mismatch.h>

template <class Policy,
          typename Container1,
          typename Container2 = Container1,
          typename BinaryPredicate = thrust::equal_to<typename Container1::value_type> >
struct Mismatch
{
  Container1 A;
  Container2 B;
  BinaryPredicate binary_pred;
  Policy policy;

  template <typename Range1, typename Range2>
  Mismatch(Policy p_, const Range1& X, const Range2& Y, BinaryPredicate binary_pred = BinaryPredicate())
    : A(X.begin(), X.end()),
      B(Y.begin(), Y.end()),
      binary_pred(binary_pred),
      policy(p_)
  {}

  void operator()(void)
  {
    thrust::mismatch(policy, A.begin(), A.end(), B.begin(), binary_pred);
  }
};


