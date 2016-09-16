#include <thrust/partition.h>

template <class Policy,
          typename Container,
          typename Predicate = thrust::identity<typename Container::value_type> >
struct Partition
{
  Container A;
  Container B; // copy of initial data
  Predicate pred;
  Policy policy;

  template <typename Range>
  Partition(Policy p_, const Range& X, Predicate pred = Predicate())
    : A(X.begin(), X.end()),
      B(X.begin(), X.end()),
      pred(pred),
      policy(p_)
  {}

  void operator()(void)
  {
    thrust::partition(policy, A.begin(), A.end(), pred);
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
          typename Container3 = Container1,
          typename Predicate = thrust::identity<typename Container1::value_type> >
struct PartitionCopy
{
  Container1 A;
  Container2 B;
  Container3 C;
  Predicate pred;
  Policy policy;

  template <typename Range1, typename Range2, typename Range3>
  PartitionCopy(Policy p_, const Range1& X, const Range2& Y, const Range3& Z, Predicate pred = Predicate())
    : A(X.begin(), X.end()),
      B(Y.begin(), Y.end()),
      C(Z.begin(), Z.end()),
      pred(pred),
      policy(p_)
  {}

  void operator()(void)
  {
    thrust::partition_copy(policy, A.begin(), A.end(), B.begin(), C.begin(), pred);
  }
};


template <class Policy,
          typename Container,
          typename Predicate = thrust::identity<typename Container::value_type> >
struct StablePartition
{
  Container A;
  Container B; // copy of initial data
  Predicate pred;
  Policy policy;

  template <typename Range>
  StablePartition(Policy p_, const Range& X, Predicate pred = Predicate())
    : A(X.begin(), X.end()),
      B(X.begin(), X.end()),
      pred(pred),
      policy(p_)
  {}

  void operator()(void)
  {
    thrust::stable_partition(policy, A.begin(), A.end(), pred);
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
          typename Container3 = Container1,
          typename Predicate = thrust::identity<typename Container1::value_type> >
struct StablePartitionCopy
{
  Container1 A;
  Container2 B;
  Container3 C;
  Predicate pred;
  Policy policy;

  template <typename Range1, typename Range2, typename Range3>
  StablePartitionCopy(Policy p_, const Range1& X, const Range2& Y, const Range3& Z, Predicate pred = Predicate())
    : A(X.begin(), X.end()),
      B(Y.begin(), Y.end()),
      C(Z.begin(), Z.end()),
      pred(pred),
      policy(p_)
  {}

  void operator()(void)
  {
    thrust::stable_partition_copy(policy, A.begin(), A.end(), B.begin(), C.begin(), pred);
  }
};


template <class Policy,
          typename Container,
          typename Predicate = thrust::identity<typename Container::value_type> >
struct IsPartitioned
{
  Container A;
  Predicate pred;
  Policy policy;

  template <typename Range>
  IsPartitioned(Policy p_, const Range& X, Predicate pred = Predicate())
    : A(X.begin(), X.end()),
      pred(pred),
      policy(p_)
  {}

  void operator()(void)
  {
    thrust::is_partitioned(policy, A.begin(), A.end(), pred);
  }
};


template <class Policy,
          typename Container,
          typename Predicate = thrust::identity<typename Container::value_type> >
struct PartitionPoint
{
  Container A;
  Predicate pred;
  Policy policy;

  template <typename Range>
  PartitionPoint(Policy p_, const Range& X, Predicate pred = Predicate())
    : A(X.begin(), X.end()),
      pred(pred),
      policy(p_)
  {}

  void operator()(void)
  {
    thrust::partition_point(policy, A.begin(), A.end(), pred);
  }
};


// is_partitioned / partition / stable_partition / partition_copy / stable_partition_copy
//template<typename InputIterator , typename OutputIterator1 , typename OutputIterator2 , typename Predicate >
//thrust::pair< OutputIterator1, 
//OutputIterator2 > 	thrust::partition_copy (InputIterator first, InputIterator last, OutputIterator1 out_true, OutputIterator2 out_false, Predicate pred)
//template<typename ForwardIterator , typename Predicate >
//ForwardIterator 	thrust::stable_partition (ForwardIterator first, ForwardIterator last, Predicate pred)
//template<typename InputIterator , typename OutputIterator1 , typename OutputIterator2 , typename Predicate >
//thrust::pair< OutputIterator1, 
//OutputIterator2 > 	thrust::stable_partition_copy (InputIterator first, InputIterator last, OutputIterator1 out_true, OutputIterator2 out_false, Predicate pred)
//template<typename ForwardIterator , typename Predicate >
//ForwardIterator 	thrust::partition_point (ForwardIterator first, ForwardIterator last, Predicate pred)
//template<typename InputIterator , typename Predicate >
//bool 	thrust::is_partitioned (InputIterator first, InputIterator last, Predicate pred)
