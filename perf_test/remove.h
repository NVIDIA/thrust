#include <thrust/remove.h>

template <class Policy,
          typename Container,
          typename T = typename Container::value_type>
struct Remove
{
  Container A;
  Container B; // copy of initial data
  T value;
  Policy policy;

  template <typename Range>
  Remove(Policy p_, const Range& X, T value)
    : A(X.begin(), X.end()),
      B(X.begin(), X.end()),
      value(value),
      policy(p_)
  {}

  void operator()(void)
  {
    thrust::remove(policy, A.begin(), A.end(), value);
  }
  
  void reset(void)
  {
    // restore initial data
    thrust::copy(policy, B.begin(), B.end(), A.begin());
  }
};

template <class Policy,
          typename Container1,
          typename Container2 = Container1,
          typename T = typename Container1::value_type>
struct RemoveCopy
{
  Container1 A;
  Container2 B;
  T value;
  Policy policy;

  template <typename Range1, typename Range2>
  RemoveCopy(Policy p_, const Range1& X, const Range2& Y, T value)
    : A(X.begin(), X.end()),
      B(Y.begin(), Y.end()),
      value(value),
      policy(p_)
  {}

  void operator()(void)
  {
    thrust::remove_copy(policy, A.begin(), A.end(), B.begin(), value);
  }
  
  void reset(void)
  {
    // restore initial data
    thrust::copy(policy, B.begin(), B.end(), A.begin());
  }
};

template <class Policy,
          typename Container1,
          typename Container2 = Container1,
          typename Predicate = thrust::identity<typename Container2::value_type> >
struct RemoveIf
{
  Container1 A, A_copy;
  Container2 B;
  Predicate pred;
  Policy policy;

  template <typename Range1, typename Range2>
  RemoveIf(Policy p_, const Range1& X, const Range2& Y, Predicate pred = Predicate())
    : A(X.begin(), X.end()), A_copy(X.begin(), X.end()),
      B(Y.begin(), Y.end()),
      pred(pred),
      policy(p_)
  {}

  void operator()(void)
  {
    thrust::remove_if(policy, A.begin(), A.end(), B.begin(), pred);
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
          typename Container3 = Container1,
          typename Predicate = thrust::identity<typename Container2::value_type> >
struct RemoveCopyIf
{
  Container1 A, A_copy;
  Container2 B;
  Container3 C;
  Predicate pred;
  Policy policy;

  template <typename Range1, typename Range2, typename Range3>
  RemoveCopyIf(Policy p_, const Range1& X, const Range2& Y, const Range3& Z, Predicate pred = Predicate())
    : A(X.begin(), X.end()), A_copy(X.begin(), X.end()),
      B(Y.begin(), Y.end()),
      C(Z.begin(), Z.end()),
      pred(pred),
      policy(p_)
  {}

  void operator()(void)
  {
    thrust::remove_copy_if(policy, A.begin(), A.end(), B.begin(), C.begin(), pred);
  }
  
  void reset(void)
  {
    // restore initial data
    thrust::copy(policy, A_copy.begin(), A_copy.end(), A.begin());
  }
};

