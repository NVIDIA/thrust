/*
 *  Copyright 2008-2009 NVIDIA Corporation
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


/*! \file unique.inl
 *  \brief Inline file for unique.h.
 */

#pragma once

#include <thrust/iterator/iterator_traits.h>
#include <thrust/transform.h>
#include <thrust/scatter.h>
#include <thrust/scan.h>
#include <thrust/functional.h>
#include <limits>
#include <thrust/detail/raw_buffer.h>

namespace thrust
{

namespace detail
{

namespace device
{

namespace detail
{

template <typename ForwardIterator, typename BinaryPredicate, typename IndexType>
ForwardIterator __unique_helper(ForwardIterator first, ForwardIterator last,
                                BinaryPredicate binary_pred)
{
    typedef typename thrust::iterator_traits<ForwardIterator>::value_type InputType;
    typedef typename thrust::iterator_traits<ForwardIterator>::difference_type difference_type;

    difference_type n = last - first;
  
    typedef raw_device_buffer<InputType> InputBuffer;
    typedef raw_device_buffer<IndexType> IndexBuffer;
    InputBuffer input(first, last);
    IndexBuffer is_first(n), scatter_to(n);

    // mark first element in each group
    thrust::binary_negate<BinaryPredicate> not_binary_pred(binary_pred);
    thrust::transform(input.begin(), input.end() - 1, input.begin() + 1, is_first.begin() + 1, not_binary_pred); 

    is_first[0] = 0; // we can ignore the first element

    // scan the predicates
    thrust::inclusive_scan(is_first.begin(), is_first.end(), scatter_to.begin());

    // scatter first elements
    thrust::scatter_if(input.begin(), input.end(), scatter_to.begin(), is_first.begin(), first);

    ForwardIterator new_last = first + scatter_to[n - 1] + 1;

    return new_last;
}

} // end namespace detail


template <typename ForwardIterator, typename BinaryPredicate>
ForwardIterator unique(ForwardIterator first, ForwardIterator last,
                       BinaryPredicate binary_pred)
{
  typedef typename thrust::iterator_traits<ForwardIterator>::difference_type difference_type;

  difference_type n = last - first;

  if(n < 2)
  {
    // ranges with length 0 and 1 are already unique
    return last;
  }

  // use 32-bit indices when possible (almost always)
  if (sizeof(difference_type) > sizeof(unsigned int) && n > std::numeric_limits<unsigned int>::max()){
      return detail::__unique_helper<ForwardIterator, BinaryPredicate, difference_type>(first, last, binary_pred);
  } else {
      return detail::__unique_helper<ForwardIterator, BinaryPredicate, unsigned int>(first, last, binary_pred);
  }
}


} // end namespace device

} // end namespace detail

} // end namespace thrust


