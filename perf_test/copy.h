#include <thrust/copy.h>

template <class Policy,
          typename Container1,
          typename Container2 = Container1>
struct Copy
{
  Container1 A;
  Container2 B;
  Policy policy;

  template <typename Range1, typename Range2>
  Copy(Policy policy, const Range1& X, const Range2& Y)
    : A(X.begin(), X.end()), B(Y.begin(), Y.end()), policy(policy)
  {}

  void operator()(void)
  {
    thrust::copy(policy, A.begin(), A.end(), B.begin());
  }
};

template <class Policy,
          typename Container1,
          typename Container2 = Container1>
struct CopyN
{
  Container1 A;
  Container2 B;
  Policy policy;

  template <typename Range1, typename Range2>
  CopyN(Policy policy, const Range1& X, const Range2& Y)
    : A(X.begin(), X.end()), B(Y.begin(), Y.end()), policy(policy)
  {}

  void operator()(void)
  {
    thrust::copy_n(policy, A.begin(), A.size(), B.begin());
  }
};

template <class Policy,
          typename Container1,
          typename Container2 = Container1,
          typename Container3 = Container1,
          typename Predicate = thrust::identity<typename Container1::value_type> >
struct CopyIf
{
  Container1 A; // values
  Container2 B; // stencil
  Container3 C; // output
  Predicate pred;
  Policy policy;

  template <typename Range1, typename Range2, typename Range3>
  CopyIf(Policy p_, const Range1& X, const Range2& Y, const Range3& Z, Predicate pred = Predicate())
    : A(X.begin(), X.end()),
      B(Y.begin(), Y.end()),
      C(Z.begin(), Z.end()),
      pred(pred), policy(p_)
  {}

  void operator()(void)
  {
    thrust::copy_if(policy, A.begin(), A.end(), B.begin(), C.begin(), pred);
  }
};

