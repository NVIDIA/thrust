/*
 *  Copyright 2008-2012 NVIDIA Corporation
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

#include <thrust/detail/config.h>
#include <thrust/detail/temporary_array.h>
#include <thrust/detail/copy.h>
#include <thrust/system/detail/internal/scalar/sort.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/distance.h>
#include <thrust/merge.h>
#include <tbb/parallel_invoke.h>

namespace thrust
{
namespace system
{
namespace tbb
{
namespace detail
{
namespace sort_detail
{

// TODO tune this based on data type and comp
const static int threshold = 128 * 1024;
  
template <typename System, typename Iterator1, typename Iterator2, typename StrictWeakOrdering>
void merge_sort(dispatchable<System> &system, Iterator1 first1, Iterator1 last1, Iterator2 first2, StrictWeakOrdering comp, bool inplace);

template <typename System, typename Iterator1, typename Iterator2, typename StrictWeakOrdering>
struct merge_sort_closure
{
  dispatchable<System> &system;
  Iterator1 first1, last1;
  Iterator2 first2;
  StrictWeakOrdering comp;
  bool inplace;

  merge_sort_closure(dispatchable<System> &system, Iterator1 first1, Iterator1 last1, Iterator2 first2, StrictWeakOrdering comp, bool inplace)
    : system(system), first1(first1), last1(last1), first2(first2), comp(comp), inplace(inplace)
  {}

  void operator()(void) const
  {
    merge_sort(system, first1, last1, first2, comp, inplace);
  }
};


template <typename System, typename Iterator1, typename Iterator2, typename StrictWeakOrdering>
void merge_sort(dispatchable<System> &system, Iterator1 first1, Iterator1 last1, Iterator2 first2, StrictWeakOrdering comp, bool inplace)
{
  typedef typename thrust::iterator_difference<Iterator1>::type difference_type;

  difference_type n = thrust::distance(first1, last1);

  if (n < threshold)
  {
    thrust::system::detail::internal::scalar::stable_sort(first1, last1, comp);
    
    if (!inplace)
      thrust::system::detail::internal::scalar::copy(first1, last1, first2);

    return;
  }

  Iterator1 mid1  = first1 + (n / 2);
  Iterator2 mid2  = first2 + (n / 2);
  Iterator2 last2 = first2 + n;

  typedef merge_sort_closure<System,Iterator1,Iterator2,StrictWeakOrdering> Closure;
  
  Closure left (system, first1, mid1,  first2, comp, !inplace);
  Closure right(system, mid1,   last1, mid2,   comp, !inplace);

  ::tbb::parallel_invoke(left, right);

  if (inplace) thrust::merge(system, first2, mid2, mid2, last2, first1, comp);
  else			   thrust::merge(system, first1, mid1, mid1, last1, first2, comp);
}

} // end namespace sort_detail


namespace sort_by_key_detail
{

// TODO tune this based on data type and comp
const static int threshold = 128 * 1024;
  
template <typename System,
          typename Iterator1,
          typename Iterator2,
          typename Iterator3,
          typename Iterator4,
          typename StrictWeakOrdering>
void merge_sort_by_key(dispatchable<System> &system,
                       Iterator1 first1,
                       Iterator1 last1,
                       Iterator2 first2,
                       Iterator3 first3,
                       Iterator4 first4,
                       StrictWeakOrdering comp,
                       bool inplace);

template <typename System,
          typename Iterator1,
          typename Iterator2,
          typename Iterator3,
          typename Iterator4,
          typename StrictWeakOrdering>
struct merge_sort_by_key_closure
{
  dispatchable<System> &system;
  Iterator1 first1, last1;
  Iterator2 first2;
  Iterator3 first3;
  Iterator4 first4;
  StrictWeakOrdering comp;
  bool inplace;

  merge_sort_by_key_closure(dispatchable<System> &system,
                            Iterator1 first1,
                            Iterator1 last1,
                            Iterator2 first2,
                            Iterator3 first3,
                            Iterator4 first4,
                            StrictWeakOrdering comp,
                            bool inplace)
    : system(system), first1(first1), last1(last1), first2(first2), first3(first3), first4(first4), comp(comp), inplace(inplace)
  {}

  void operator()(void) const
  {
    merge_sort_by_key(system, first1, last1, first2, first3, first4, comp, inplace);
  }
};


template <typename System,
          typename Iterator1,
          typename Iterator2,
          typename Iterator3,
          typename Iterator4,
          typename StrictWeakOrdering>
void merge_sort_by_key(dispatchable<System> &system,
                       Iterator1 first1,
                       Iterator1 last1,
                       Iterator2 first2,
                       Iterator3 first3,
                       Iterator4 first4,
                       StrictWeakOrdering comp,
                       bool inplace)
{
  typedef typename thrust::iterator_difference<Iterator1>::type difference_type;

  difference_type n = thrust::distance(first1, last1);
  
  Iterator1 mid1  = first1 + (n / 2);
  Iterator2 mid2  = first2 + (n / 2);
  Iterator3 mid3  = first3 + (n / 2);
  Iterator4 mid4  = first4 + (n / 2);
  Iterator2 last2 = first2 + n;
  Iterator3 last3 = first3 + n;

  if (n < threshold)
  {
    thrust::system::detail::internal::scalar::stable_sort_by_key(first1, last1, first2, comp);
    
    if (!inplace)
    {
      thrust::system::detail::internal::scalar::copy(first1, last1, first3);
      thrust::system::detail::internal::scalar::copy(first2, last2, first4);
    }

    return;
  }

  typedef merge_sort_by_key_closure<System,Iterator1,Iterator2,Iterator3,Iterator4,StrictWeakOrdering> Closure;
  
  Closure left (system, first1, mid1,  first2, first3, first4, comp, !inplace);
  Closure right(system, mid1,   last1, mid2,   mid3,   mid4,   comp, !inplace);

  ::tbb::parallel_invoke(left, right);

  // TODO replace with thrust::merge_by_key
  if(inplace)
  {
    //thrust::system::tbb::detail::merge_by_key(system, first3, mid3, mid3, last3, first4, mid4, first1, first2, comp);
    thrust::merge_by_key(system, first3, mid3, mid3, last3, first4, mid4, first1, first2, comp);
  }
  else
  {
    //thrust::system::tbb::detail::merge_by_key(system, first1, mid1, mid1, last1, first2, mid2, first3, first4, comp);
    thrust::merge_by_key(system, first1, mid1, mid1, last1, first2, mid2, first3, first4, comp);
  }
}

} // end namespace sort_detail

template<typename System,
         typename RandomAccessIterator,
         typename StrictWeakOrdering>
void stable_sort(dispatchable<System> &system,
                 RandomAccessIterator first,
                 RandomAccessIterator last,
                 StrictWeakOrdering comp)
{
  typedef typename thrust::iterator_value<RandomAccessIterator>::type key_type;

  thrust::detail::temporary_array<key_type, System> temp(system, first, last);

  sort_detail::merge_sort(system, first, last, temp.begin(), comp, true);
}

template<typename System,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename StrictWeakOrdering>
  void stable_sort_by_key(dispatchable<System> &system,
                          RandomAccessIterator1 first1,
                          RandomAccessIterator1 last1,
                          RandomAccessIterator2 first2,
                          StrictWeakOrdering comp)
{
  typedef typename thrust::iterator_value<RandomAccessIterator1>::type key_type;
  typedef typename thrust::iterator_value<RandomAccessIterator2>::type val_type;

  RandomAccessIterator2 last2 = first2 + thrust::distance(first1, last1);

  thrust::detail::temporary_array<key_type, System> temp1(system, first1, last1);
  thrust::detail::temporary_array<val_type, System> temp2(system, first2, last2);

  sort_by_key_detail::merge_sort_by_key(system, first1, last1, first2, temp1.begin(), temp2.begin(), comp, true);
}

} // end namespace detail
} // end namespace tbb
} // end namespace system
} // end namespace thrust

