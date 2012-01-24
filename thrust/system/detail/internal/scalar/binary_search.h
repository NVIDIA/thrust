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


/*! \file binary_search.h
 *  \brief Sequential implementation of binary search algorithms.
 */

#pragma once

#include <thrust/advance.h>
#include <thrust/distance.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/detail/function.h>

namespace thrust
{
namespace system
{
namespace detail
{
namespace internal
{
namespace scalar
{

template <typename ForwardIterator,
          typename T,
          typename StrictWeakOrdering>
ForwardIterator lower_bound(ForwardIterator first,
                            ForwardIterator last,
                            const T& val,
                            StrictWeakOrdering comp)
{
  // wrap comp
  thrust::detail::host_function<
    StrictWeakOrdering,
    bool
  > wrapped_comp(comp);

  typedef typename thrust::iterator_difference<ForwardIterator>::type difference_type;

  difference_type len = thrust::distance(first, last);

  while(len > 0)
  {
    difference_type half = len >> 1;
    ForwardIterator middle = first;

    thrust::advance(middle, half);

    if(wrapped_comp(*middle, val))
    {
      first = middle;
      ++first;
      len = len - half - 1;
    }
    else
    {
      len = half;
    }
  }

  return first;
}


template <typename ForwardIterator, typename T, typename StrictWeakOrdering>
ForwardIterator upper_bound(ForwardIterator first,
                            ForwardIterator last,
                            const T& val, 
                            StrictWeakOrdering comp)
{
  // wrap comp
  thrust::detail::host_function<
    StrictWeakOrdering,
    bool
  > wrapped_comp(comp);

  typedef typename thrust::iterator_difference<ForwardIterator>::type difference_type;

  difference_type len = thrust::distance(first, last);

  while(len > 0)
  {
    difference_type half = len >> 1;
    ForwardIterator middle = first;

    thrust::advance(middle, half);

    if(wrapped_comp(val, *middle))
    {
      len = half;
    }
    else
    {
      first = middle;
      ++first;
      len = len - half - 1;
    }
  }

  return first;
}

template <typename ForwardIterator,
          typename T,
          typename StrictWeakOrdering>
bool binary_search(ForwardIterator first,
                   ForwardIterator last,
                   const T& val, 
                   StrictWeakOrdering comp)
{
  ForwardIterator iter = thrust::system::detail::internal::scalar::lower_bound(first, last, val, comp);

  // wrap comp
  thrust::detail::host_function<
    StrictWeakOrdering,
    bool
  > wrapped_comp(comp);

  return iter != last && !wrapped_comp(val,*iter);
}

} // end namespace scalar
} // end namespace internal
} // end namespace detail
} // end namespace system
} // end namespace thrust

