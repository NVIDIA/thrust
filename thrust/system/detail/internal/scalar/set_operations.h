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


/*! \file set_operations.h
 *  \brief Sequential implementation of set operation functions.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/system/detail/internal/scalar/copy.h>
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

template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename StrictWeakOrdering>
  OutputIterator set_difference(InputIterator1 first1,
                                InputIterator1 last1,
                                InputIterator2 first2,
                                InputIterator2 last2,
                                OutputIterator result,
                                StrictWeakOrdering comp)
{
  // wrap comp
  thrust::detail::host_function<
    StrictWeakOrdering,
    bool
  > wrapped_comp(comp);

  while(first1 != last1 && first2 != last2)
  {
    if(wrapped_comp(*first1,*first2))
    {
      *result = *first1;
      ++first1;
      ++result;
    } // end if
    else if(wrapped_comp(*first2,*first1))
    {
      ++first2;
    } // end else if
    else
    {
      ++first1;
      ++first2;
    } // end else
  } // end while

  return scalar::copy(first1, last1, result);
} // end set_difference()


template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename StrictWeakOrdering>
  OutputIterator set_intersection(InputIterator1 first1,
                                  InputIterator1 last1,
                                  InputIterator2 first2,
                                  InputIterator2 last2,
                                  OutputIterator result,
                                  StrictWeakOrdering comp)
{
  // wrap comp
  thrust::detail::host_function<
    StrictWeakOrdering,
    bool
  > wrapped_comp(comp);

  while(first1 != last1 && first2 != last2)
  {
    if(wrapped_comp(*first1,*first2))
    {
      ++first1;
    } // end if
    else if(wrapped_comp(*first2,*first1))
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
  OutputIterator set_symmetric_difference(InputIterator1 first1,
                                          InputIterator1 last1,
                                          InputIterator2 first2,
                                          InputIterator2 last2,
                                          OutputIterator result,
                                          StrictWeakOrdering comp)
{
  // wrap comp
  thrust::detail::host_function<
    StrictWeakOrdering,
    bool
  > wrapped_comp(comp);

  while(first1 != last1 && first2 != last2)
  {
    if(wrapped_comp(*first1,*first2))
    {
      *result = *first1;
      ++first1;
      ++result;
    } // end if
    else if(wrapped_comp(*first2,*first1))
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

  return scalar::copy(first2, last2, scalar::copy(first1, last1, result));
} // end set_symmetric_difference()


template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename StrictWeakOrdering>
  OutputIterator set_union(InputIterator1 first1,
                           InputIterator1 last1,
                           InputIterator2 first2,
                           InputIterator2 last2,
                           OutputIterator result,
                           StrictWeakOrdering comp)
{
  // wrap comp
  thrust::detail::host_function<
    StrictWeakOrdering,
    bool
  > wrapped_comp(comp);

  while(first1 != last1 && first2 != last2)
  {
    if(wrapped_comp(*first1,*first2))
    {
      *result = *first1;
      ++first1;
    } // end if
    else if(wrapped_comp(*first2,*first1))
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

  return scalar::copy(first2, last2, scalar::copy(first1, last1, result));
} // end set_union()

} // end namespace scalar
} // end namespace internal
} // end namespace detail
} // end namespace system
} // end namespace thrust

