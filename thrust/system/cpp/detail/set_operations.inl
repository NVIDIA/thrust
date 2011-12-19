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

/*
 *
 * Copyright (c) 1994
 * Hewlett-Packard Company
 *
 * Permission to use, copy, modify, distribute and sell this software
 * and its documentation for any purpose is hereby granted without fee,
 * provided that the above copyright notice appear in all copies and
 * that both that copyright notice and this permission notice appear
 * in supporting documentation.  Hewlett-Packard Company makes no
 * representations about the suitability of this software for any
 * purpose.  It is provided "as is" without express or implied warranty.
 *
 *
 * Copyright (c) 1996
 * Silicon Graphics Computer Systems, Inc.
 *
 * Permission to use, copy, modify, distribute and sell this software
 * and its documentation for any purpose is hereby granted without fee,
 * provided that the above copyright notice appear in all copies and
 * that both that copyright notice and this permission notice appear
 * in supporting documentation.  Silicon Graphics makes no
 * representations about the suitability of this software for any
 * purpose.  It is provided "as is" without express or implied warranty.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/system/cpp/detail/set_operations.h>
#include <thrust/detail/copy.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/detail/wrapped_function.h>

namespace thrust
{
namespace system
{
namespace cpp
{
namespace detail
{


template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename StrictWeakOrdering>
  OutputIterator set_difference(tag,
                                InputIterator1 first1,
                                InputIterator1 last1,
                                InputIterator2 first2,
                                InputIterator2 last2,
                                OutputIterator result,
                                StrictWeakOrdering comp)
{
  // wrap comp twice
  // the difference is the order of the InputIterators' references
  thrust::detail::host_wrapped_binary_function<
    StrictWeakOrdering,
    typename thrust::iterator_reference<InputIterator1>::type,
    typename thrust::iterator_reference<InputIterator2>::type,
    bool
  > wrapped_comp1(comp);

  thrust::detail::host_wrapped_binary_function<
    StrictWeakOrdering,
    typename thrust::iterator_reference<InputIterator2>::type,
    typename thrust::iterator_reference<InputIterator1>::type,
    bool
  > wrapped_comp2(comp);

  while(first1 != last1 && first2 != last2)
  {
    if(wrapped_comp1(*first1, *first2))
    {
      *result = *first1;
      ++first1;
      ++result;
    } // end if
    else if(wrapped_comp2(*first2, *first1))
    {
      ++first2;
    } // end else if
    else
    {
      ++first1;
      ++first2;
    } // end else
  } // end while

  return thrust::copy(first1, last1, result);
} // end set_difference()


template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename StrictWeakOrdering>
  OutputIterator set_intersection(tag,
                                  InputIterator1 first1,
                                  InputIterator1 last1,
                                  InputIterator2 first2,
                                  InputIterator2 last2,
                                  OutputIterator result,
                                  StrictWeakOrdering comp)
{
  // wrap comp twice
  // the difference is the order of the InputIterators' references
  thrust::detail::host_wrapped_binary_function<
    StrictWeakOrdering,
    typename thrust::iterator_reference<InputIterator1>::type,
    typename thrust::iterator_reference<InputIterator2>::type,
    bool
  > wrapped_comp1(comp);

  thrust::detail::host_wrapped_binary_function<
    StrictWeakOrdering,
    typename thrust::iterator_reference<InputIterator2>::type,
    typename thrust::iterator_reference<InputIterator1>::type,
    bool
  > wrapped_comp2(comp);

  while(first1 != last1 && first2 != last2)
  {
    if(wrapped_comp1(*first1, *first2))
    {
      ++first1;
    } // end if
    else if(wrapped_comp2(*first2, *first1))
    {
      ++first2;
    } // end else if
    else
    {
      *result = *first1;
      ++first1;
      ++first2;
      ++result;
    } // end else
  } // end while

  return result;
} // end set_intersection()


template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename StrictWeakOrdering>
  OutputIterator set_symmetric_difference(tag,
                                          InputIterator1 first1,
                                          InputIterator1 last1,
                                          InputIterator2 first2,
                                          InputIterator2 last2,
                                          OutputIterator result,
                                          StrictWeakOrdering comp)
{
  // wrap comp twice
  // the difference is the order of the InputIterators' references
  thrust::detail::host_wrapped_binary_function<
    StrictWeakOrdering,
    typename thrust::iterator_reference<InputIterator1>::type,
    typename thrust::iterator_reference<InputIterator2>::type,
    bool
  > wrapped_comp1(comp);

  thrust::detail::host_wrapped_binary_function<
    StrictWeakOrdering,
    typename thrust::iterator_reference<InputIterator2>::type,
    typename thrust::iterator_reference<InputIterator1>::type,
    bool
  > wrapped_comp2(comp);

  while(first1 != last1 && first2 != last2)
  {
    if(wrapped_comp1(*first1, *first2))
    {
      *result = *first1;
      ++first1;
      ++result;
    } // end if
    else if(wrapped_comp2(*first2, *first1))
    {
      *result = *first2;
      ++first2;
      ++result;
    } // end else if
    else
    {
      ++first1;
      ++first2;
    } // end else
  } // end while

  return thrust::copy(first2, last2, thrust::copy(first1, last1, result));
} // end set_symmetric_difference()


template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename StrictWeakOrdering>
  OutputIterator set_union(tag,
                           InputIterator1 first1,
                           InputIterator1 last1,
                           InputIterator2 first2,
                           InputIterator2 last2,
                           OutputIterator result,
                           StrictWeakOrdering comp)
{
  // wrap comp twice
  // the difference is the order of the InputIterators' references
  thrust::detail::host_wrapped_binary_function<
    StrictWeakOrdering,
    typename thrust::iterator_reference<InputIterator1>::type,
    typename thrust::iterator_reference<InputIterator2>::type,
    bool
  > wrapped_comp1(comp);

  thrust::detail::host_wrapped_binary_function<
    StrictWeakOrdering,
    typename thrust::iterator_reference<InputIterator2>::type,
    typename thrust::iterator_reference<InputIterator1>::type,
    bool
  > wrapped_comp2(comp);

  while(first1 != last1 && first2 != last2)
  {
    if(wrapped_comp1(*first1, *first2))
    {
      *result = *first1;
      ++first1;
    } // end if
    else if(wrapped_comp2(*first2, *first1))
    {
      *result = *first2;
      ++first2;
    } // end else if
    else
    {
      *result = *first1;
      ++first1;
      ++first2;
    } // end else

    ++result;
  } // end while

  return thrust::copy(first2, last2, thrust::copy(first1, last1, result));
} // end set_union()


} // end detail
} // end cpp
} // end system
} // end thrust

