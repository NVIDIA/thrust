#include <thrust/tabulate.h>
#include <thrust/functional.h>

template <class Policy,
          typename Container,
          typename UnaryFunction = thrust::negate<typename Container::value_type> >
struct Tabulate
{
  Container A;
  UnaryFunction unary_op;
  Policy policy;

  template <typename Range>
  Tabulate(Policy p_, const Range& X,
           UnaryFunction unary_op = UnaryFunction())
    : A(X.begin(), X.end()),
      unary_op(unary_op),
      policy(p_)
  {}

  void operator()(void)
  {
    thrust::tabulate(policy, A.begin(), A.end(), unary_op);
  }
};


