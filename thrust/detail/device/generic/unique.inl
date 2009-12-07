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
namespace generic
{
namespace detail
{

template <typename IndexType,
          typename InputIterator,
          typename OutputIterator,
          typename BinaryPredicate>
OutputIterator unique_copy(InputIterator first,
                           InputIterator last,
                           OutputIterator output,
                           BinaryPredicate binary_pred)
{
    typedef typename thrust::iterator_traits<InputIterator>::value_type InputType;
    typedef typename thrust::iterator_space<OutputIterator>::type Space;
    
    typedef raw_buffer<IndexType,Space> IndexBuffer;

    IndexType n = last - first;
  
    IndexBuffer is_first(n), scatter_to(n);

    // mark first element in each group
    thrust::binary_negate<BinaryPredicate> not_binary_pred(binary_pred);
    thrust::transform(first, last - 1, first + 1, is_first.begin() + 1, not_binary_pred); 

    // ignore the first element for now 
    is_first[0] = 0; 
    
    // scan the predicates
    thrust::inclusive_scan(is_first.begin(), is_first.end(), scatter_to.begin());
    
    // mark the first element before scatter 
    is_first[0] = 1;
    
    // scatter first elements
    thrust::scatter_if(first, last, scatter_to.begin(), is_first.begin(), output);
    
    OutputIterator new_last = output + scatter_to[n - 1] + 1;

    return new_last;
}

} // end namespace detail


template <typename ForwardIterator,
          typename BinaryPredicate>
ForwardIterator unique(ForwardIterator first,
                       ForwardIterator last,
                       BinaryPredicate binary_pred)
{
    typedef typename thrust::iterator_traits<ForwardIterator>::value_type InputType;
    typedef typename thrust::iterator_space<ForwardIterator>::type        Space;

    typedef raw_buffer<InputType,Space> InputBuffer;
    InputBuffer input(first, last);

    return thrust::detail::device::generic::unique_copy(input.begin(), input.end(), first, binary_pred);
}

template <typename InputIterator,
          typename OutputIterator,
          typename BinaryPredicate>
OutputIterator unique_copy(InputIterator first,
                           InputIterator last,
                           OutputIterator output,
                           BinaryPredicate binary_pred)
{
  typedef typename thrust::iterator_traits<InputIterator>::difference_type difference_type;

  difference_type n = last - first;

  // ranges with length 0 and 1 are already unique
  if(n < 2)
    return output + n;  

  // use 32-bit indices when possible (almost always)
  if (sizeof(difference_type) > sizeof(unsigned int) && n > std::numeric_limits<unsigned int>::max())
      return detail::unique_copy<difference_type>(first, last, output, binary_pred);
  else
      return detail::unique_copy<unsigned int>   (first, last, output, binary_pred);
}

} // end namespace generic
} // end namespace device
} // end namespace detail
} // end namespace thrust

