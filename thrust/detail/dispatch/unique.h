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


/*! \file unique.h
 *  \brief Defines the interface to
 *         the dispatch layer of the
 *         unique function.
 */

#pragma once

#include <thrust/iterator/iterator_categories.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/transform.h>
#include <thrust/scatter.h>
#include <thrust/scan.h>
#include <thrust/functional.h>
#include <algorithm>
#include <limits>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

namespace thrust
{

namespace detail
{

namespace dispatch
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
  
    thrust::device_ptr<InputType> input      = thrust::device_malloc<InputType>(n);
    thrust::device_ptr<IndexType> is_first   = thrust::device_malloc<IndexType>(n);
    thrust::device_ptr<IndexType> scatter_to = thrust::device_malloc<IndexType>(n);
   
    thrust::copy(first, last, input);

    // mark first element in each group
    thrust::binary_negate<BinaryPredicate> not_binary_pred(binary_pred);
    thrust::transform(input, input + (n - 1), input + 1, is_first + 1, not_binary_pred); 

    is_first[0] = 0; // we can ignore the first element

    // scan the predicates
    thrust::inclusive_scan(is_first, is_first + n, scatter_to);

    // scatter first elements
    thrust::scatter_if(input, input + n, scatter_to, is_first, first);

    ForwardIterator new_last = first + scatter_to[n - 1] + 1;

    thrust::device_free(input);
    thrust::device_free(is_first);
    thrust::device_free(scatter_to);

    return new_last;
}

} // end namespace detail


///////////////
// Host Path //
///////////////
template <typename ForwardIterator, typename BinaryPredicate>
ForwardIterator unique(ForwardIterator first, ForwardIterator last,
                       BinaryPredicate binary_pred,
                       thrust::forward_host_iterator_tag)
{
  return std::unique(first, last, binary_pred);
}


/////////////////
// Device Path //
/////////////////
template <typename ForwardIterator, typename BinaryPredicate>
ForwardIterator unique(ForwardIterator first, ForwardIterator last,
                       BinaryPredicate binary_pred,
                       thrust::random_access_device_iterator_tag)
{
  typedef typename thrust::iterator_traits<ForwardIterator>::value_type      InputType;
  typedef typename thrust::iterator_traits<ForwardIterator>::difference_type difference_type;

  difference_type n = last - first;

  if(n < 2 )
  {
    // ranges with length 0 and 1 are already unique
    return last;
  }


  // use unsigned int for scatter indices when possible
  if (sizeof(difference_type) > sizeof(unsigned int) && n > std::numeric_limits<unsigned int>::max()){
      return detail::__unique_helper<ForwardIterator, BinaryPredicate, difference_type>(first, last, binary_pred);
  } else {
      return detail::__unique_helper<ForwardIterator, BinaryPredicate, unsigned int>(first, last, binary_pred);
  }
}


} // end namespace dispatch

} // end namespace detail

} // end namespace thrust

