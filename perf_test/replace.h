#include <thrust/replace.h>

template <class Policy,
          typename Container,
          typename T = typename Container::value_type>
struct Replace
{
  Container A, A_copy;
  T old_value, new_value;
  Policy policy;

  template <typename Range>
  Replace(Policy p_, const Range& X, const T& old_value, const T& new_value)
    : A(X.begin(), X.end()), A_copy(X.begin(), X.end()),
      old_value(old_value), new_value(new_value),
      policy(p_)
  {}

  void operator()(void)
  {
    thrust::replace(policy, A.begin(), A.end(), old_value, new_value);
  }
  
  void reset(void)
  {
    // restore initial data
    thrust::copy(policy, A_copy.begin(), A_copy.end(), A.begin());
  }
};

template <class Policy,
          typename Container1,
          typename Container2 = Container1,
          typename Predicate = thrust::identity<typename Container2::value_type>,
          typename T = typename Container1::value_type>
struct ReplaceIf
{
  Container1 A, A_copy;
  Container2 B;
  Predicate pred;
  T new_value;
  Policy policy;

  template <typename Range1, typename Range2>
  ReplaceIf(Policy p_, const Range1& X, const Range2& Y, Predicate pred, const T& new_value)
    : A(X.begin(), X.end()), A_copy(X.begin(), X.end()),
      B(Y.begin(), Y.end()),
      pred(pred), new_value(new_value),
      policy(p_)
  {}

  void operator()(void)
  {
    thrust::replace_if(policy, A.begin(), A.end(), B.begin(), pred, new_value);
  }
  
  void reset(void)
  {
    // restore initial data
    thrust::copy(policy, A_copy.begin(), A_copy.end(), A.begin());
  }
};

template <class Policy,
          typename Container1,
          typename Container2 = Container1,
          typename T = typename Container1::value_type>
struct ReplaceCopy
{
  Container1 A;
  Container2 B;
  T old_value, new_value;
  Policy policy;

  template <typename Range1, typename Range2>
  ReplaceCopy(Policy p_, const Range1& X, const Range2& Y, const T& old_value, const T& new_value)
    : A(X.begin(), X.end()),
      B(Y.begin(), Y.end()),
      old_value(old_value), new_value(new_value),
      policy(p_)
  {}

  void operator()(void)
  {
    thrust::replace_copy(policy, A.begin(), A.end(), B.begin(), old_value, new_value);
  }
};

template <class Policy,
          typename Container1,
          typename Container2 = Container1,
          typename Container3 = Container1,
          typename Predicate = thrust::identity<typename Container2::value_type>,
          typename T = typename Container1::value_type>
struct ReplaceCopyIf
{
  Container1 A, A_copy; // input
  Container2 B;         // stencil
  Container3 C;         // output
  Predicate pred;
  T new_value;
  Policy policy;

  template <typename Range1, typename Range2, typename Range3>
  ReplaceCopyIf(Policy p_, const Range1& X, const Range2& Y, const Range3& Z, Predicate pred, const T& new_value)
    : A(X.begin(), X.end()), A_copy(X.begin(), X.end()),
      B(Y.begin(), Y.end()),
      C(Z.begin(), Z.end()),
      pred(pred), new_value(new_value),
      policy(p_)
  {}

  void operator()(void)
  {
    thrust::replace_copy_if(policy, A.begin(), A.end(), B.begin(), C.begin(), pred, new_value);
  }
};


