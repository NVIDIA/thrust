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
#include <thrust/system/detail/internal/scalar/insertion_sort.h>
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

static int threshold = 64;
  
template <typename Iterator1, typename Iterator2, typename StrictWeakOrdering>
void merge_sort(Iterator1 first1, Iterator1 last1, Iterator2 first2, StrictWeakOrdering comp, bool inplace);

template <typename Iterator1, typename Iterator2, typename StrictWeakOrdering>
struct merge_sort_closure
{
  Iterator1 first1, last1;
  Iterator2 first2;
  StrictWeakOrdering comp;
  bool inplace;

  merge_sort_closure(Iterator1 first1, Iterator1 last1, Iterator2 first2, StrictWeakOrdering comp, bool inplace)
    : first1(first1), last1(last1), first2(first2), comp(comp), inplace(inplace)
  {}

  void operator()(void) const
  {
    merge_sort(first1, last1, first2, comp, inplace);
  }
};


template <typename Iterator1, typename Iterator2, typename StrictWeakOrdering>
void merge_sort(Iterator1 first1, Iterator1 last1, Iterator2 first2, StrictWeakOrdering comp, bool inplace)
{
  typedef typename thrust::iterator_difference<Iterator1>::type difference_type;

  difference_type n = thrust::distance(first1, last1);

  if (n < threshold)
  {
    thrust::system::detail::internal::scalar::insertion_sort(first1, last1, comp);
    
    if (!inplace)
      thrust::copy(first1, last1, first2); // XXX replace with trivial sequential copy

    return;
  }

  Iterator1 mid1  = first1 + (n / 2);
  Iterator2 mid2  = first2 + (n / 2);
  Iterator2 last2 = first2 + n;

  typedef merge_sort_closure<Iterator1,Iterator2,StrictWeakOrdering> Closure;
  
  Closure left (first1, mid1,  first2, comp, !inplace);
  Closure right(mid1,   last1, mid2,   comp, !inplace);

  ::tbb::parallel_invoke(left, right);

  if (inplace) thrust::merge(first2, mid2, mid2, last2, first1, comp);
  else			   thrust::merge(first1, mid1, mid1, last1, first2, comp);
}

} // end namespace sort_detail

template<typename RandomAccessIterator,
         typename StrictWeakOrdering>
void stable_sort(tag,
                 RandomAccessIterator first,
                 RandomAccessIterator last,
                 StrictWeakOrdering comp)
{
  typedef typename thrust::iterator_system<RandomAccessIterator>::type system;
  typedef typename thrust::iterator_value<RandomAccessIterator>::type key_type;

  thrust::detail::temporary_array<key_type, system> temp(first, last);

  sort_detail::merge_sort(first, last, temp.begin(), comp, true);
}

} // end namespace detail
} // end namespace tbb
} // end namespace system
} // end namespace thrust

