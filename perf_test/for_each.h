#include <thrust/for_each.h>

struct default_for_each_function
{
  template <typename T>
  __host__ __device__
  void operator()(T& x)
  {
    x = T();
  }
};

template <class Policy,
          typename Container,
          typename UnaryFunction = default_for_each_function>
struct ForEach
{
  Container A;
  UnaryFunction unary_op;
  Policy policy;

  template <typename Range>
  ForEach(Policy policy_, const Range& X, UnaryFunction unary_op = UnaryFunction())
    : A(X.begin(), X.end()),
      unary_op(unary_op), policy(policy_)
  {}

  void operator()(void)
  {
    thrust::for_each(policy, A.begin(), A.end(), unary_op);
  }
};

