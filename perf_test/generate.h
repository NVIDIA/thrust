#include <thrust/generate.h>

template <typename T>
struct default_generate_function
{
  __host__ __device__
  T operator()(void)
  {
    return T();
  }
};

template <class Policy,
          typename Container,
          typename UnaryFunction = default_generate_function<typename Container::value_type> >
struct Generate
{
  Container A;
  UnaryFunction unary_op;
  Policy policy;

  template <typename Range>
  Generate(Policy policy_, const Range& X, UnaryFunction unary_op = UnaryFunction())
    : A(X.begin(), X.end()),
      unary_op(unary_op),
      policy(policy_)
  {}

  void operator()(void)
  {
    thrust::generate(policy, A.begin(), A.end(), unary_op);
  }
};

template <class Policy,
          typename Container,
          typename UnaryFunction = default_generate_function<typename Container::value_type> >
struct GenerateN
{
  Container A;
  UnaryFunction unary_op;
  Policy policy;

  template <typename Range>
  GenerateN(Policy policy_, const Range& X, UnaryFunction unary_op = UnaryFunction())
    : A(X.begin(), X.end()),
      unary_op(unary_op),
      policy(policy_)
  {}

  void operator()(void)
  {
    thrust::generate_n(policy, A.begin(), A.size(), unary_op);
  }
};

