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


/*! \file binary_search.h
 *  \brief C++ implementation of binary search algorithms.
 */

#pragma once

#include <thrust/advance.h>
#include <thrust/distance.h>
#include <thrust/iterator/iterator_traits.h>

#include <thrust/binary_search.h>
#include <thrust/detail/wrapped_function.h>
#include <thrust/system/cpp/detail/tag.h>

// TODO replace the code below with calls to thrust::detail::backend::generic::scalar::*
//      when warnings about __host__ calling __host__ __device__ are silenceable

namespace thrust
{
namespace system
{
namespace cpp
{
namespace detail
{

template <typename ForwardIterator, typename T, typename StrictWeakOrdering>
ForwardIterator lower_bound(tag,
                            ForwardIterator first,
                            ForwardIterator last,
                            const T& val,
                            StrictWeakOrdering comp)
{
  typedef typename thrust::iterator_difference<ForwardIterator>::type difference_type;

  // wrap comp
  thrust::detail::host_wrapped_binary_function<
    StrictWeakOrdering,
    typename thrust::iterator_reference<ForwardIterator>::type,
    const T&,
    bool
  > wrapped_comp(comp);

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
ForwardIterator upper_bound(tag,
                            ForwardIterator first,
                            ForwardIterator last,
                            const T& val, 
                            StrictWeakOrdering comp)
{
  typedef typename thrust::iterator_difference<ForwardIterator>::type difference_type;

  // wrap comp
  thrust::detail::host_wrapped_binary_function<
    StrictWeakOrdering,
    const T&,
    typename thrust::iterator_reference<ForwardIterator>::type,
    bool
  > wrapped_comp(comp);

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

template <typename ForwardIterator, typename T, typename StrictWeakOrdering>
bool binary_search(tag,
                   ForwardIterator first,
                   ForwardIterator last,
                   const T& val, 
                   StrictWeakOrdering comp)
{
  ForwardIterator iter = thrust::lower_bound(first,last,val,comp);

  // wrap comp
  thrust::detail::host_wrapped_binary_function<
    StrictWeakOrdering,
    const T&,
    typename thrust::iterator_reference<ForwardIterator>::type,
    bool
  > wrapped_comp(comp);

  return iter != last && !wrapped_comp(val, *iter);
}

} // end namespace detail
} // end namespace cpp
} // end namespace system
} // end namespace thrust

