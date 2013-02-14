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


/*! \file partition.h
 *  \brief Sequential implementations of partition functions.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/pair.h>
#include <thrust/detail/temporary_array.h>
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

template <typename ForwardIterator1,
          typename ForwardIterator2>
void iter_swap(ForwardIterator1 iter1, ForwardIterator2 iter2)
{
  // XXX this isn't correct because it doesn't use thrust::swap
  using namespace thrust::detail;

  typedef typename thrust::iterator_value<ForwardIterator1>::type T;

  T temp = *iter1;
  *iter1 = *iter2;
  *iter2 = temp;
}

template<typename ForwardIterator,
         typename Predicate>
  ForwardIterator partition(ForwardIterator first,
                            ForwardIterator last,
                            Predicate pred)
{
  if (first == last)
    return first;

  // wrap pred
  thrust::detail::host_function<
    Predicate,
    bool
  > wrapped_pred(pred);

  while (wrapped_pred(*first))
  {
    if (++first == last)
      return first;
  }

  ForwardIterator next = first;

  while (++next != last)
  {
    if (wrapped_pred(*next))
    {
      iter_swap(first, next);
      ++first;
    }
  }

  return first;
}

template<typename ForwardIterator,
         typename Predicate>
  ForwardIterator stable_partition(ForwardIterator first,
                                   ForwardIterator last,
                                   Predicate pred)
{
  // wrap pred
  thrust::detail::host_function<
    Predicate,
    bool
  > wrapped_pred(pred);

  // XXX the type of exec should be:
  //     typedef decltype(select_system(first, last)) system;
  typedef typename thrust::iterator_system<ForwardIterator>::type ExecutionPolicy;
  typedef typename thrust::iterator_value<ForwardIterator>::type T;

  typedef thrust::detail::temporary_array<T,ExecutionPolicy> TempRange;
  typedef typename TempRange::iterator                       TempIterator;

  // XXX presumes ExecutionPolicy is default constructible
  ExecutionPolicy exec;
  TempRange temp(exec, first, last);

  for(TempIterator iter = temp.begin(); iter != temp.end(); ++iter)
  {
    if (wrapped_pred(*iter))
    {
      *first = *iter;
      ++first;
    }
  }

  ForwardIterator middle = first;

  for(TempIterator iter = temp.begin(); iter != temp.end(); ++iter)
  {
    if (!wrapped_pred(*iter))
    {
      *first = *iter;
      ++first;
    }
  }

  return middle;
}

template<typename ForwardIterator,
         typename InputIterator,
         typename Predicate>
  ForwardIterator stable_partition(ForwardIterator first,
                                   ForwardIterator last,
                                   InputIterator stencil,
                                   Predicate pred)
{
  // wrap pred
  thrust::detail::host_function<
    Predicate,
    bool
  > wrapped_pred(pred);

  // XXX the type of exec should be:
  //     typedef decltype(select_system(first, stencil)) system;
  typedef typename thrust::iterator_system<ForwardIterator>::type ExecutionPolicy;
  typedef typename thrust::iterator_value<ForwardIterator>::type T;

  typedef thrust::detail::temporary_array<T,ExecutionPolicy> TempRange;
  typedef typename TempRange::iterator                       TempIterator;

  // XXX presumes ExecutionPolicy is default constructible
  ExecutionPolicy exec;
  TempRange temp(exec, first, last);

  InputIterator stencil_iter = stencil;
  for(TempIterator iter = temp.begin(); iter != temp.end(); ++iter, ++stencil_iter)
  {
    if (wrapped_pred(*stencil_iter))
    {
      *first = *iter;
      ++first;
    }
  }

  ForwardIterator middle = first;
  stencil_iter = stencil;

  for(TempIterator iter = temp.begin(); iter != temp.end(); ++iter, ++stencil_iter)
  {
    if (!wrapped_pred(*stencil_iter))
    {
      *first = *iter;
      ++first;
    }
  }

  return middle;
}

template<typename InputIterator,
         typename OutputIterator1,
         typename OutputIterator2,
         typename Predicate>
  thrust::pair<OutputIterator1,OutputIterator2>
    stable_partition_copy(InputIterator first,
                          InputIterator last,
                          OutputIterator1 out_true,
                          OutputIterator2 out_false,
                          Predicate pred)
{
  // wrap pred
  thrust::detail::host_function<
    Predicate,
    bool
  > wrapped_pred(pred);

  for(; first != last; ++first)
  {
    if(wrapped_pred(*first))
    {
      *out_true = *first;
      ++out_true;
    } // end if
    else
    {
      *out_false = *first;
      ++out_false;
    } // end else
  }

  return thrust::make_pair(out_true, out_false);
}

template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator1,
         typename OutputIterator2,
         typename Predicate>
  thrust::pair<OutputIterator1,OutputIterator2>
    stable_partition_copy(InputIterator1 first,
                          InputIterator1 last,
                          InputIterator2 stencil,
                          OutputIterator1 out_true,
                          OutputIterator2 out_false,
                          Predicate pred)
{
  // wrap pred
  thrust::detail::host_function<
    Predicate,
    bool
  > wrapped_pred(pred);

  for(; first != last; ++first, ++stencil)
  {
    if(wrapped_pred(*stencil))
    {
      *out_true = *first;
      ++out_true;
    } // end if
    else
    {
      *out_false = *first;
      ++out_false;
    } // end else
  }

  return thrust::make_pair(out_true, out_false);
}

} // end namespace scalar
} // end namespace internal
} // end namespace detail
} // end namespace system
} // end namespace thrust

