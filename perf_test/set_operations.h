#include <thrust/set_operations.h>

#include <thrust/sort.h>

template <class Policy,
          typename Container1,
          typename Container2 = Container1,
          typename Container3 = Container1,
          typename StrictWeakCompare = thrust::less<typename Container1::value_type> >
struct SetDifference
{
  Container1 A;
  Container2 B;
  Container3 C;
  StrictWeakCompare comp;
  Policy policy;

  template <typename Range1, typename Range2, typename Range3>
  SetDifference(Policy p_, const Range1& X, const Range2& Y, const Range3& Z, StrictWeakCompare comp = StrictWeakCompare())
    : A(X.begin(), X.end()),
      B(Y.begin(), Y.end()),
      C(Z.begin(), Z.end()),
      comp(comp),
      policy(p_)
  {
    thrust::stable_sort(policy, A.begin(), A.end(), comp);
    thrust::stable_sort(policy, B.begin(), B.end(), comp);
  }

  void operator()(void)
  {
    size_t size = thrust::set_difference(policy, A.begin(), A.end(), B.begin(), B.end(), C.begin(), comp) - C.begin();
#ifdef _PRINT
    static bool print = true;
#else
    static bool print = false;
#endif
    if (print)
    {
      printf("diff= %d\n", (int)size);
      print = false;
    }
  }
};

template <class Policy,
          typename Container1,
          typename Container2 = Container1,
          typename Container3 = Container1,
          typename StrictWeakCompare = thrust::less<typename Container1::value_type> >
struct SetIntersection
{
  Container1 A;
  Container2 B;
  Container3 C;
  StrictWeakCompare comp;
  Policy policy;

  template <typename Range1, typename Range2, typename Range3>
  SetIntersection(Policy p_, const Range1& X, const Range2& Y, const Range3& Z, StrictWeakCompare comp = StrictWeakCompare())
    : A(X.begin(), X.end()),
      B(Y.begin(), Y.end()),
      C(Z.begin(), Z.end()),
      comp(comp),
      policy(p_)
  {
    thrust::stable_sort(policy, A.begin(), A.end(), comp);
    thrust::stable_sort(policy, B.begin(), B.end(), comp);
  }

  void operator()(void)
  {
    size_t size = thrust::set_intersection(policy, A.begin(), A.end(), B.begin(), B.end(), C.begin(), comp) - C.begin();
#ifdef _PRINT
    static bool print = true;
#else
    static bool print = false;
#endif
    if (print)
    {
      printf("inter= %d\n", (int)size);
      print = false;
    }
  }
};

template <class Policy,
          typename Container1,
          typename Container2 = Container1,
          typename Container3 = Container1,
          typename StrictWeakCompare = thrust::less<typename Container1::value_type> >
struct SetSymmetricDifference
{
  Container1 A;
  Container2 B;
  Container3 C;
  StrictWeakCompare comp;
  Policy policy;

  template <typename Range1, typename Range2, typename Range3>
  SetSymmetricDifference(Policy p_, const Range1& X, const Range2& Y, const Range3& Z, StrictWeakCompare comp = StrictWeakCompare())
    : A(X.begin(), X.end()),
      B(Y.begin(), Y.end()),
      C(Z.begin(), Z.end()),
      comp(comp),
      policy(p_)
  {
    thrust::stable_sort(policy, A.begin(), A.end(), comp);
    thrust::stable_sort(policy, B.begin(), B.end(), comp);
  }

  void operator()(void)
  {
    size_t size = thrust::set_symmetric_difference(policy, A.begin(), A.end(), B.begin(), B.end(), C.begin(), comp) - C.begin();
#ifdef _PRINT
    static bool print = true;
#else
    static bool print = false;
#endif
    if (print)
    {
      printf("sym_dif= %d\n", (int)size);
      print = false;
    }
  }
};

template <class Policy,
          typename Container1,
          typename Container2 = Container1,
          typename Container3 = Container1,
          typename StrictWeakCompare = thrust::less<typename Container1::value_type> >
struct SetUnion
{
  Container1 A;
  Container2 B;
  Container3 C;
  StrictWeakCompare comp;
  Policy policy;

  template <typename Range1, typename Range2, typename Range3>
  SetUnion(Policy p_, const Range1& X, const Range2& Y, const Range3& Z, StrictWeakCompare comp = StrictWeakCompare())
    : A(X.begin(), X.end()),
      B(Y.begin(), Y.end()),
      C(Z.begin(), Z.end()),
      comp(comp),
      policy(p_)
  {
    thrust::stable_sort(policy, A.begin(), A.end(), comp);
    thrust::stable_sort(policy, B.begin(), B.end(), comp);
  }

  void operator()(void)
  {
    size_t  size = thrust::set_union(policy, A.begin(), A.end(), B.begin(), B.end(), C.begin(), comp) - C.begin();
#ifdef _PRINT
    static bool print = true;
#else
    static bool print = false;
#endif
    if (print)
    {
      printf("union= %d\n", (int)size);
      print = false;
    }
  }
};

