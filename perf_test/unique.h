#include <thrust/unique.h>

template <class Policy,
          typename Container,
          typename BinaryPredicate = thrust::equal_to<typename Container::value_type> >
struct Unique
{
  Container A, A_copy;
  BinaryPredicate pred;
  Policy policy;

  template <typename Range>
  Unique(Policy p_, const Range& X, BinaryPredicate pred = BinaryPredicate())
    : A(X.begin(), X.end()), A_copy(X.begin(), X.end()),
      pred(pred),
      policy(p_)
  {}

  void operator()(void)
  {
    thrust::unique(policy, A.begin(), A.end(), pred);
  }

  void reset(void)
  {
    thrust::copy(policy, A_copy.begin(), A_copy.end(), A.begin());
  }
};

template <class Policy,
          typename Container1,
          typename Container2 = Container1,
          typename BinaryPredicate = thrust::equal_to<typename Container1::value_type> >
struct UniqueCopy
{
  Container1 A;
  Container2 B;
  BinaryPredicate pred;
  Policy policy;

  template <typename Range1, typename Range2>
  UniqueCopy(Policy p_, const Range1& X, const Range2& Y, BinaryPredicate pred = BinaryPredicate())
    : A(X.begin(), X.end()),
      B(Y.begin(), Y.end()),
      pred(pred),
      policy(p_)
  {}

  void operator()(void)
  {
    thrust::unique_copy(policy, A.begin(), A.end(), B.begin(), pred);
  }
};

template <class Policy,
          typename Container1,
          typename Container2 = Container1,
          typename BinaryPredicate = thrust::equal_to<typename Container1::value_type> >
struct UniqueByKey
{
  Container1 A, A_copy; // keys
  Container2 B, B_copy; // values
  BinaryPredicate pred;
  Policy policy;

  template <typename Range1, typename Range2>
  UniqueByKey(Policy p_, const Range1& X, const Range2& Y, BinaryPredicate pred = BinaryPredicate())
    : A(X.begin(), X.end()), A_copy(X.begin(), X.end()),
      B(Y.begin(), Y.end()), B_copy(Y.begin(), Y.end()),
      pred(pred),
      policy(p_)
  {}

  void operator()(void)
  {
    thrust::unique_by_key(policy, A.begin(), A.end(), B.begin(), pred);
  }

  void reset(void)
  {
    thrust::copy(policy, A_copy.begin(), A_copy.end(), A.begin());
    thrust::copy(policy, B_copy.begin(), B_copy.end(), B.begin());
  }
};

template <class Policy,
          typename Container1,
          typename Container2 = Container1,
          typename Container3 = Container1,
          typename Container4 = Container2,
          typename BinaryPredicate = thrust::equal_to<typename Container1::value_type> >
struct UniqueByKeyCopy
{
  Container1 A; // input keys
  Container2 B; // input values
  Container3 C; // output keys
  Container4 D; // output values
  BinaryPredicate pred;
  Policy policy;

  template <typename Range1, typename Range2, typename Range3, typename Range4>
  UniqueByKeyCopy(Policy p_, const Range1& X, const Range2& Y, const Range3& Z, const Range4& W, BinaryPredicate pred = BinaryPredicate())
    : A(X.begin(), X.end()),
      B(Y.begin(), Y.end()),
      C(Z.begin(), Z.end()),
      D(W.begin(), W.end()),
      pred(pred),
      policy(p_)
  {}

  void operator()(void)
  {
    thrust::unique_by_key_copy(policy, A.begin(), A.end(), B.begin(), C.begin(), D.begin(), pred);
  }
};

