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


/*! \file unique.inl
 *  \brief Inline file for unique.h.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/system/detail/generic/unique.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/transform.h>
#include <thrust/unique.h>
#include <thrust/detail/temporary_array.h>
#include <thrust/detail/internal_functional.h>
#include <thrust/detail/copy_if.h>
#include <thrust/distance.h>
#include <thrust/functional.h>

namespace thrust
{
namespace system
{
namespace detail
{
namespace generic
{


template<typename ForwardIterator>
  ForwardIterator unique(tag,
                         ForwardIterator first,
                         ForwardIterator last)
{
  typedef typename thrust::iterator_traits<ForwardIterator>::value_type InputType;

  return thrust::unique(first, last, thrust::equal_to<InputType>());
} // end unique()


template<typename ForwardIterator,
         typename BinaryPredicate>
  ForwardIterator unique(tag,
                         ForwardIterator first,
                         ForwardIterator last,
                         BinaryPredicate binary_pred)
{
  typedef typename thrust::iterator_traits<ForwardIterator>::value_type InputType;
  typedef typename thrust::iterator_system<ForwardIterator>::type        System;
  
  thrust::detail::temporary_array<InputType,System> input(first, last);
  
  return thrust::unique_copy(input.begin(), input.end(), first, binary_pred);
} // end unique()


template<typename InputIterator,
         typename OutputIterator>
  OutputIterator unique_copy(tag,
                             InputIterator first,
                             InputIterator last,
                             OutputIterator output)
{
  typedef typename thrust::iterator_value<InputIterator>::type value_type;
  return thrust::unique_copy(first,last,output,thrust::equal_to<value_type>());
} // end unique_copy()


template<typename InputIterator,
         typename OutputIterator,
         typename BinaryPredicate>
  OutputIterator unique_copy(tag,
                             InputIterator first,
                             InputIterator last,
                             OutputIterator output,
                             BinaryPredicate binary_pred)
{
  typedef typename thrust::detail::minimum_system<
    typename thrust::iterator_system<InputIterator>::type,
    typename thrust::iterator_system<OutputIterator>::type
  >::type System;
  
  // empty sequence
  if(first == last)
    return output;
  
  thrust::detail::temporary_array<int,System> stencil(thrust::distance(first, last));
  
  // mark first element in each group
  stencil[0] = 1; 
  thrust::transform(first, last - 1, first + 1, stencil.begin() + 1, thrust::detail::not2(binary_pred)); 
  
  return thrust::copy_if(first, last, stencil.begin(), output, thrust::identity<int>());
} // end unique_copy()


} // end namespace generic
} // end namespace detail
} // end namespace system
} // end namespace thrust

