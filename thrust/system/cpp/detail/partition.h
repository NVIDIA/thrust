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


/*! \file partition.h
 *  \brief cpp implementations of partition functions
 */

#pragma once

#include <thrust/pair.h>
#include <thrust/detail/temporary_array.h>
#include <thrust/system/cpp/detail/tag.h>
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

template <typename ForwardIterator1,
          typename ForwardIterator2>
void iter_swap(ForwardIterator1 iter1, ForwardIterator2 iter2)
{
  using namespace thrust::detail;

  typedef typename thrust::iterator_value<ForwardIterator1>::type T;

  T temp = *iter1;
  *iter1 = *iter2;
  *iter2 = temp;
}

template<typename ForwardIterator,
         typename Predicate>
  ForwardIterator partition(tag,
                            ForwardIterator first,
                            ForwardIterator last,
                            Predicate pred)
{
  if (first == last)
    return first;

  // wrap pred
  thrust::detail::host_wrapped_unary_function<
    Predicate,
    typename thrust::iterator_reference<ForwardIterator>::type,
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
  ForwardIterator stable_partition(tag,
                                   ForwardIterator first,
                                   ForwardIterator last,
                                   Predicate pred)
{
  typedef typename thrust::iterator_value<ForwardIterator>::type T;
  typedef thrust::detail::temporary_array<T,thrust::cpp::tag> TempRange;
  typedef typename TempRange::iterator                        TempIterator;

  // wrap pred
  thrust::detail::host_wrapped_unary_function<
    Predicate,
    typename thrust::iterator_reference<TempIterator>::type,
    bool
  > wrapped_pred(pred);

  TempRange temp(first, last);

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
    if (!bool(wrapped_pred(*iter)))
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
    stable_partition_copy(tag,
                          InputIterator first,
                          InputIterator last,
                          OutputIterator1 out_true,
                          OutputIterator2 out_false,
                          Predicate pred)
{
  // wrap pred
  thrust::detail::host_wrapped_unary_function<
    Predicate,
    typename thrust::iterator_reference<InputIterator>::type,
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


} // end namespace detail
} // end namespace cpp
} // end namespace system
} // end namespace thrust

