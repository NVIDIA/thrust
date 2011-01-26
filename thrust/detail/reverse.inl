/*
 *  Copyright 2008-2011 NVIDIA Corporation
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


/*! \file reverse.inl
 *  \brief Inline file for reverse.h.
 */

#include <thrust/detail/config.h>

#include <thrust/advance.h>
#include <thrust/distance.h>
#include <thrust/copy.h>
#include <thrust/swap.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/reverse_iterator.h>

namespace thrust
{

template<typename BidirectionalIterator>
  void reverse(BidirectionalIterator first,
               BidirectionalIterator last)
{
  typedef typename thrust::iterator_difference<BidirectionalIterator>::type difference_type;

  // find the midpoint of [first,last)
  difference_type N = thrust::distance(first, last);
  BidirectionalIterator mid(first);
  thrust::advance(mid, N / 2);

  // swap elements of [first,mid) with [last - 1, mid)
  thrust::swap_ranges(first, mid, thrust::make_reverse_iterator(last));
} // end reverse()

template<typename BidirectionalIterator,
         typename OutputIterator>
  OutputIterator reverse_copy(BidirectionalIterator first,
                              BidirectionalIterator last,
                              OutputIterator result)
{
  return thrust::copy(thrust::make_reverse_iterator(last),
                      thrust::make_reverse_iterator(first),
                      result);
} // end reverse_copy()

} // end thrust

