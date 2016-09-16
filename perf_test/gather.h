#include <thrust/gather.h>

template <class Policy,
          typename Container1,
          typename Container2 = Container1,
          typename Container3 = Container2>
struct Gather
{
  Container1 A; // map
  Container2 B; // source
  Container3 C; // output
  Policy policy;

  template <typename Range1, typename Range2, typename Range3>
  Gather(Policy policy_, const Range1& X, const Range2& Y, const Range3& Z)
    : A(X.begin(), X.end()),
      B(Y.begin(), Y.end()),
      C(Z.begin(), Z.end()),
      policy(policy_)
  {}

  void operator()(void)
  {
    thrust::gather(policy, A.begin(), A.end(), B.begin(), C.begin());
  }
};

template <class Policy,
          typename Container1,
          typename Container2 = Container1,
          typename Container3 = Container1,
          typename Container4 = Container2,
          typename Predicate = thrust::identity<typename Container2::value_type> >
struct GatherIf
{
  Container1 A; // map
  Container2 B; // stencil
  Container3 C; // source
  Container4 D; // output
  Predicate pred;
  Policy policy;

  template <typename Range1, typename Range2, typename Range3, typename Range4>
  GatherIf(Policy policy_, const Range1& X, const Range2& Y, const Range3& Z, const Range4& W, Predicate pred = Predicate())
    : A(X.begin(), X.end()),
      B(Y.begin(), Y.end()),
      C(Z.begin(), Z.end()),
      D(W.begin(), W.end()),
      pred(pred),
      policy(policy_)
  {}

  void operator()(void)
  {
    thrust::gather_if(policy, A.begin(), A.end(), B.begin(), C.begin(), D.begin(), pred);
  }
};

