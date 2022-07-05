/*
 *  Copyright 2022 NVIDIA Corporation
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

#pragma once

#include <thrust/detail/config.h>
#include <thrust/system/detail/sequential/execution_policy.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/copy.h>
#include <thrust/swap_ranges.h>

THRUST_NAMESPACE_BEGIN
namespace system
{
namespace detail
{
namespace sequential
{


template <typename DerivedPolicy, typename ForwardIterator>
__host__ __device__
  ForwardIterator shift_left(sequential::execution_policy<DerivedPolicy> &,
                             ForwardIterator first,
                             ForwardIterator last,
                             typename thrust::iterator_traits<ForwardIterator>::difference_type n)
{
  if (n <= 0) {
    return last;
  }

  ForwardIterator current = first;
  while (n-- != 0) {
    if (current == last) {
      return first;
    }
    ++current;
  }

  while (current != last) {
    *first = *current;
  }

  return first;
}


template <typename DerivedPolicy, typename ForwardIterator>
__host__ __device__
  ForwardIterator shift_right(sequential::execution_policy<DerivedPolicy> &,
                             ForwardIterator first,
                             ForwardIterator last,
                             typename thrust::iterator_traits<ForwardIterator>::difference_type n)
{
  if (n <= 0) {
    return first;
  }

  ForwardIterator current = first;
  for (auto i = n; i != 0; --i, ++current) {
    if (current == last) {
      return last;
    }
  }

  // First check whether we can just move parts of [first, current) into
  // [current, last) and be done with it
  ForwardIterator trail = first;
  ForwardIterator lead  = current;
  for (; trail != current; ++trail, ++lead) {
    if (lead == last) {
      thrust::copy(first, trail, current);
      return current;
    }
  }

  // [first, current) and [trail, lead) are now equal sized subranges of [first, last)
  // swap [first, current) and [trail, lead) until we reach the end
  while (true) {
    trail = thrust::swap_ranges(first, current, trail);

    // check whether we reached end
    ForwardIterator mid = first;
    for (auto i = n; i != 0; --i, ++lead, ++mid) {
      if (lead == last) {
        thrust::copy(first, mid, trail);
        return current;
      }
    }
  }
}

} // end namespace sequential
} // end namespace detail
} // end namespace system
THRUST_NAMESPACE_END
