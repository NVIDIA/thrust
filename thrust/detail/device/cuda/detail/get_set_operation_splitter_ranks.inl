/*
 *  Copyright 2008-2010 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#include <thrust/detail/device/cuda/detail/get_set_operation_splitter_ranks.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/tuple.h>
#include <thrust/detail/device/generic/scalar/select.h>
#include <thrust/binary_search.h>
#include <thrust/iterator/iterator_traits.h>

namespace thrust
{
namespace detail
{
namespace device
{
namespace cuda
{
namespace detail
{

namespace get_set_operation_splitter_ranks_detail
{

template<typename T>
struct mult_by
  : thrust::unary_function<T,T>
{
  T _value;
  
  mult_by(const T& v):_value(v){}
  
  __host__ __device__
  T operator()(const T& v) const
  {
    return _value * v;
  }
};

// this predicate tests two two-element tuples
// we first use a Compare for the first element
// if the first elements are equivalent, we use
// < for the second elements
template<typename Compare>
  struct compare_first_less_second
{
  compare_first_less_second(Compare c)
    : comp(c) {}

  template<typename Tuple>
  __host__ __device__
  bool operator()(Tuple lhs, Tuple rhs)
  {
    return comp(lhs.get<0>(), rhs.get<0>()) || (!comp(rhs.get<0>(), lhs.get<0>()) && lhs.get<1>() < rhs.get<1>());
  }

  Compare comp;
}; // end compare_first_less_second

template<typename Iterator1, typename Iterator2, typename Compare>
  struct select_functor
{
  Iterator1 first1, last1;
  Iterator2 first2, last2;
  Compare comp;

  select_functor(Iterator1 f1, Iterator1 l1,
                 Iterator2 f2, Iterator2 l2,
                 Compare c)
    : first1(f1), last1(l1), first2(f2), last2(l2), comp(c)
  {}
  
  // satisfy AdaptableUnaryFunction
  typedef typename thrust::iterator_value<Iterator1>::type      result_type;
  typedef typename thrust::iterator_difference<Iterator1>::type argument_type;

  __host__ __device__
  result_type operator()(argument_type k)
  {
    typedef typename thrust::iterator_value<Iterator1>::type value_type;
    return thrust::detail::device::generic::scalar::select(first1, last1, first2, last2, k, comp);
  }
}; // end select_functor

template<typename Iterator1, typename Iterator2, typename Compare>
  class merge_iterator
{
  typedef thrust::counting_iterator<typename thrust::iterator_difference<Iterator1>::type> counting_iterator;
  typedef select_functor<Iterator1,Iterator2,Compare> function;

  public:
    typedef thrust::transform_iterator<function, counting_iterator> type;
}; // end merge_iterator

template<typename Iterator1, typename Iterator2, typename Compare>
  typename merge_iterator<Iterator1,Iterator2,Compare>::type
    make_merge_iterator(Iterator1 first1, Iterator1 last1,
                        Iterator2 first2, Iterator2 last2,
                        Compare comp)
{
  typedef typename thrust::iterator_difference<Iterator1>::type difference;
  difference zero = 0;

  select_functor<Iterator1,Iterator2,Compare> f(first1,last1,first2,last2,comp);
  return thrust::make_transform_iterator(thrust::make_counting_iterator<difference>(zero),
                                         f);
} // end make_merge_iterator()

template<typename Integer>
  class leapfrog_iterator
{
  typedef thrust::counting_iterator<Integer> counter;

  public:
    typedef thrust::transform_iterator<mult_by<Integer>, counter> type;
}; // end leapfrog_iterator

template<typename Integer>
  typename leapfrog_iterator<Integer>::type
    make_leapfrog_iterator(Integer init, Integer leap_size)
{
  return thrust::make_transform_iterator(thrust::make_counting_iterator<Integer>(init),
                                         mult_by<Integer>(leap_size));
} // end make_leapfrog_iterator()


template<typename RandomAccessIterator>
  class splitter_iterator
{
  typedef typename thrust::iterator_difference<RandomAccessIterator>::type difference;
  typedef typename leapfrog_iterator<difference>::type leapfrog_iterator;

  public:
    typedef thrust::permutation_iterator<RandomAccessIterator, leapfrog_iterator> type;
}; // end splitter_iterator

template<typename RandomAccessIterator, typename Size>
  typename splitter_iterator<RandomAccessIterator>::type
    make_splitter_iterator(RandomAccessIterator iter, Size split_size)
{
  typedef typename thrust::iterator_difference<RandomAccessIterator>::type difference;
  return thrust::make_permutation_iterator(iter, make_leapfrog_iterator<difference>(0, split_size));
} // end make_splitter_iterator()


template<typename Compare>
  struct strong_compare
{
  strong_compare(Compare c)
    : comp(c) {}

  // T1 and T2 are tuples
  template<typename T1, typename T2>
  __host__ __device__
  bool operator()(T1 lhs, T2 rhs)
  {
    if(comp(lhs.get<0>(), rhs.get<0>()))
    {
      return true;
    }

    return lhs.get<1>() < rhs.get<1>();
  }

  Compare comp;
};

} // end get_set_operation_splitter_ranks_detail

template<typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename RandomAccessIterator3,
         typename RandomAccessIterator4,
         typename Compare,
         typename Size1,
         typename Size2>
  void get_set_operation_splitter_ranks(RandomAccessIterator1 first1,
                                        RandomAccessIterator1 last1,
                                        RandomAccessIterator2 first2,
                                        RandomAccessIterator2 last2,
                                        RandomAccessIterator3 splitter_ranks1,
                                        RandomAccessIterator4 splitter_ranks2,
                                        Compare comp,
                                        Size1 partition_size,
                                        Size2 num_splitters_from_each_range)
{
  using namespace get_set_operation_splitter_ranks_detail;

  typedef typename thrust::iterator_difference<RandomAccessIterator1>::type difference1;
  typedef typename thrust::iterator_difference<RandomAccessIterator2>::type difference2;

  const difference1 num_elements1 = last1 - first1;
  const difference2 num_elements2 = last2 - first2;

  // zip up the ranges with a counter to disambiguate repeated elements during rank-finding
  typedef thrust::tuple<RandomAccessIterator1,thrust::counting_iterator<difference1> > iterator_tuple1;
  typedef thrust::tuple<RandomAccessIterator2,thrust::counting_iterator<difference2> > iterator_tuple2;
  typedef thrust::zip_iterator<iterator_tuple1> iterator_and_counter1;
  typedef thrust::zip_iterator<iterator_tuple2> iterator_and_counter2;

  iterator_and_counter1 first_and_counter1 =
    thrust::make_zip_iterator(thrust::make_tuple(first1, thrust::make_counting_iterator<difference1>(0)));
  iterator_and_counter1 last_and_counter1 = first_and_counter1 + num_elements1;

  // make the second range begin counting at num_elements1 so they sort after elements from the first range when ambiguous
  iterator_and_counter2 first_and_counter2 =
    thrust::make_zip_iterator(thrust::make_tuple(first2, thrust::make_counting_iterator<difference2>(num_elements1)));
  iterator_and_counter2 last_and_counter2 = first_and_counter2 + num_elements2;

  // create the range [first1[partition_size], first1[2*partition_size], first1[3*partition_size], ...]
  typedef typename splitter_iterator<iterator_and_counter1>::type splitter_iterator1;

  // we +1 to begin at first1[partition_size] instead of first1[0]
  splitter_iterator1 splitters1_begin = make_splitter_iterator(first_and_counter1, partition_size) + 1;
  splitter_iterator1 splitters1_end = splitters1_begin + num_splitters_from_each_range;

  // create the range [first2[partition_size], first2[2*partition_size], first2[3*partition_size], ...]
  typedef typename splitter_iterator<iterator_and_counter1>::type splitter_iterator2;

  // we +1 to begin at first2[partition_size] instead of first1[0]
  splitter_iterator2 splitters2_begin = make_splitter_iterator(first_and_counter2, partition_size) + 1;
  splitter_iterator2 splitters2_end = splitters2_begin + num_splitters_from_each_range;

  typedef compare_first_less_second<Compare> splitter_compare;

  typedef typename merge_iterator<splitter_iterator1,splitter_iterator2,splitter_compare>::type merge_iterator;

  // "merge" the splitters
  merge_iterator splitters_begin = make_merge_iterator(splitters1_begin, splitters1_end,
                                                       splitters2_begin, splitters2_end,
                                                       splitter_compare(comp));
  merge_iterator splitters_end   = splitters_begin + 2 * num_splitters_from_each_range;

  // find the rank of each splitter in the other range
  thrust::lower_bound(first_and_counter2, last_and_counter2,
                      splitters_begin, splitters_end, 
                      splitter_ranks2, strong_compare<Compare>(comp));

  thrust::upper_bound(first_and_counter1, last_and_counter1,
                      splitters_begin, splitters_end,
                      splitter_ranks1, strong_compare<Compare>(comp));
} // end get_set_operation_splitter_ranks()

} // end detail
} // end cuda
} // end device
} // end detail
} // end thrust

