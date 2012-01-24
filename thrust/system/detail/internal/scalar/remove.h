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


/*! \file remove.h
 *  \brief Sequential implementations of remove functions.
 */

#pragma once

#include <thrust/detail/config.h>
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

template<typename ForwardIterator,
         typename Predicate>
  ForwardIterator remove_if(ForwardIterator first,
                            ForwardIterator last,
                            Predicate pred)
{
  // wrap pred
  thrust::detail::host_function<
    Predicate,
    bool
  > wrapped_pred(pred);

  // advance iterators until wrapped_pred(*first) is true or we reach the end of input
  while(first != last && !wrapped_pred(*first))
    ++first;

  if(first == last)
    return first;

  // result always trails first 
  ForwardIterator result = first;

  ++first;

  while(first != last)
  {
    if(!wrapped_pred(*first))
    {
      *result = *first;
      ++result;
    }
    ++first;
  }

  return result;
}


template<typename ForwardIterator,
         typename InputIterator,
         typename Predicate>
  ForwardIterator remove_if(ForwardIterator first,
                            ForwardIterator last,
                            InputIterator stencil,
                            Predicate pred)
{
  // wrap pred
  thrust::detail::host_function<
    Predicate,
    bool
  > wrapped_pred(pred);

  // advance iterators until wrapped_pred(*stencil) is true or we reach the end of input
  while(first != last && !wrapped_pred(*stencil))
  {
    ++first;
    ++stencil;
  }

  if(first == last)
    return first;

  // result always trails first 
  ForwardIterator result = first;

  ++first;
  ++stencil;

  while(first != last)
  {
    if(!wrapped_pred(*stencil))
    {
      *result = *first;
      ++result;
    }
    ++first;
    ++stencil;
  }

  return result;
}


template<typename InputIterator,
         typename OutputIterator,
         typename Predicate>
  OutputIterator remove_copy_if(InputIterator first,
                                InputIterator last,
                                OutputIterator result,
                                Predicate pred)
{
  // wrap pred
  thrust::detail::host_function<
    Predicate,
    bool
  > wrapped_pred(pred);

  while (first != last)
  {
    if (!wrapped_pred(*first))
    {
      *result = *first;
      ++result;
    }

    ++first;
  }

  return result;
}

template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename Predicate>
  OutputIterator remove_copy_if(InputIterator1 first,
                                InputIterator1 last,
                                InputIterator2 stencil,
                                OutputIterator result,
                                Predicate pred)
{
  // wrap pred
  thrust::detail::host_function<
    Predicate,
    bool
  > wrapped_pred(pred);

  while (first != last)
  {
    if (!wrapped_pred(*stencil))
    {
      *result = *first;
      ++result;
    }

    ++first;
    ++stencil;
  }

  return result;
}

} // end namespace scalar
} // end namespace internal
} // end namespace detail
} // end namespace system
} // end namespace thrust

