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


/*! \file min_max_element.inl
 *  \brief Inline file for min_element.h and max_element.h.
 */

#pragma once

#include <thrust/iterator/iterator_traits.h>
#include <thrust/functional.h>
#include <thrust/extrema.h>
#include <thrust/detail/dispatch/min_max_element.h>


namespace thrust
{


template <typename ForwardIterator>
ForwardIterator min_element(ForwardIterator first, ForwardIterator last)
{
  // use < predicate
  typedef typename thrust::iterator_traits<ForwardIterator>::value_type InputType;
  return thrust::min_element(first, last, thrust::less<InputType>());
} // end min_element()


template <typename ForwardIterator, typename BinaryPredicate>
ForwardIterator min_element(ForwardIterator first, ForwardIterator last,
                            BinaryPredicate comp)
{
  // dispatch on category
  return thrust::detail::dispatch::min_element(first, last, comp,
    typename thrust::iterator_traits<ForwardIterator>::iterator_category());
} // end min_element()


template <typename ForwardIterator>
ForwardIterator max_element(ForwardIterator first, ForwardIterator last)
{
  // use < predicate
  typedef typename thrust::iterator_traits<ForwardIterator>::value_type InputType;
  return thrust::max_element(first, last, thrust::less<InputType>());
} // end max_element()


template <typename ForwardIterator, typename BinaryPredicate>
ForwardIterator max_element(ForwardIterator first, ForwardIterator last,
                            BinaryPredicate comp)
{
  // dispatch on category
  return thrust::detail::dispatch::max_element(first, last, comp,
    typename thrust::iterator_traits<ForwardIterator>::iterator_category());
} // end max_element()


} // end namespace thrust

