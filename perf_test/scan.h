#include <thrust/scan.h>

template <class Policy,
          typename Container1,
          typename Container2 = Container1,
          typename BinaryFunction = thrust::plus<typename Container1::value_type> >
struct InclusiveScan
{
  Container1 A;
  Container2 B;
  BinaryFunction binary_op;
  Policy policy;

  template <typename Range1, typename Range2>
  InclusiveScan(Policy p_, const Range1& X, const Range2& Y,
                BinaryFunction binary_op = BinaryFunction())
    : A(X.begin(), X.end()),
      B(Y.begin(), Y.end()),
      binary_op(binary_op),
      policy(p_)
  {}

  void operator()(void)
  {
    thrust::inclusive_scan(policy, A.begin(), A.end(), B.begin(), binary_op);
  }
};

template <class Policy,
          typename Container1,
          typename Container2 = Container1,
          typename T = typename Container1::value_type,
          typename BinaryFunction = thrust::plus<T> >
struct ExclusiveScan
{
  Container1 A;
  Container2 B;
  T init;
  BinaryFunction binary_op;
  Policy policy;

  template <typename Range1, typename Range2>
  ExclusiveScan(Policy p_, const Range1& X, const Range2& Y,
                T init = T(0),
                BinaryFunction binary_op = BinaryFunction())
    : A(X.begin(), X.end()),
      B(Y.begin(), Y.end()),
      init(init),
      binary_op(binary_op),
      policy(p_)
  {}

  void operator()(void)
  {
    thrust::exclusive_scan(policy, A.begin(), A.end(), B.begin(), init, binary_op);
  }
};

template <class Policy,
          typename Container1,
          typename Container2 = Container1,
          typename Container3 = Container2,
          typename BinaryPredicate = thrust::equal_to<typename Container1::value_type>,
          typename BinaryFunction = thrust::plus<typename Container2::value_type> >
struct InclusiveScanByKey
{
  Container1 A;
  Container2 B;
  Container3 C;
  BinaryPredicate binary_pred;
  BinaryFunction binary_op;
  Policy policy;

  template <typename Range1, typename Range2, typename Range3>
  InclusiveScanByKey(Policy p_, const Range1& X, const Range2& Y, const Range3& Z,
                     BinaryPredicate binary_pred = BinaryPredicate(),
                     BinaryFunction binary_op = BinaryFunction())
    : A(X.begin(), X.end()),
      B(Y.begin(), Y.end()),
      C(Z.begin(), Z.end()),
      binary_pred(binary_pred),
      binary_op(binary_op),
      policy(p_)
  {}

  void operator()(void)
  {
    thrust::inclusive_scan_by_key(policy, A.begin(), A.end(), B.begin(), C.begin(), binary_pred, binary_op);
  }
};

template <class Policy,
          typename Container1,
          typename Container2 = Container1,
          typename Container3 = Container2,
          typename T = typename Container2::value_type,
          typename BinaryPredicate = thrust::equal_to<typename Container1::value_type>,
          typename BinaryFunction = thrust::plus<T> >
struct ExclusiveScanByKey
{
  Container1 A;
  Container2 B;
  Container3 C;
  T init;
  BinaryPredicate binary_pred;
  BinaryFunction binary_op;
  Policy policy;

  template <typename Range1, typename Range2, typename Range3>
  ExclusiveScanByKey(Policy p_, const Range1& X, const Range2& Y, const Range3& Z,
                     T init = T(0),
                     BinaryPredicate binary_pred = BinaryPredicate(),
                     BinaryFunction binary_op = BinaryFunction())
    : A(X.begin(), X.end()),
      B(Y.begin(), Y.end()),
      C(Z.begin(), Z.end()),
      init(init),
      binary_pred(binary_pred),
      binary_op(binary_op),
      policy(p_)
  {}

  void operator()(void)
  {
    thrust::exclusive_scan_by_key(policy, A.begin(), A.end(), B.begin(), C.begin(), init, binary_pred, binary_op);
  }
};


