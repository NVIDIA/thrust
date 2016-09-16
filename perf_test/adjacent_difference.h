#include <thrust/adjacent_difference.h>

template <class Policy,
          typename Container1,
          typename Container2     = Container1,
          typename BinaryFunction = thrust::minus<typename Container1::value_type> >
struct AdjacentDifference
{
  Policy policy;
  Container1 A;
  Container2 B;
  BinaryFunction binary_op;

  template <typename Range1, typename Range2>
  AdjacentDifference(Policy         policy,
                     const Range1&  X,
                     const Range2&  Y,
                     BinaryFunction binary_op = BinaryFunction())
      : policy(policy),
        A(X.begin(), X.end()),
        B(Y.begin(), Y.end()),
        binary_op(binary_op)
  {}

  void operator()(void)
  {
    thrust::adjacent_difference(policy, A.begin(), A.end(), B.begin(), binary_op);
  }
};

