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
#include <thrust/detail/backend/dereference.h>

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

  T temp = backend::dereference(iter1);
  backend::dereference(iter1) = backend::dereference(iter2);
  backend::dereference(iter2) =  temp;
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

  while (pred(thrust::detail::backend::dereference(first)))
  {
    if (++first == last)
      return first;
  }

  ForwardIterator next = first;

  while (++next != last)
  {
    if (pred(thrust::detail::backend::dereference(next)))
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

  TempRange temp(first, last);

  for(TempIterator iter = temp.begin(); iter != temp.end(); ++iter)
  {
    if (pred(thrust::detail::backend::dereference(iter)))
    {
      thrust::detail::backend::dereference(first) = thrust::detail::backend::dereference(iter);
      ++first;
    }
  }

  ForwardIterator middle = first;

  for(TempIterator iter = temp.begin(); iter != temp.end(); ++iter)
  {
    if (!bool(pred(thrust::detail::backend::dereference(iter))))
    {
      thrust::detail::backend::dereference(first) = thrust::detail::backend::dereference(iter);
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
  for(; first != last; ++first)
  {
    if(pred(thrust::detail::backend::dereference(first)))
    {
      thrust::detail::backend::dereference(out_true) = thrust::detail::backend::dereference(first);
      ++out_true;
    } // end if
    else
    {
      thrust::detail::backend::dereference(out_false) = thrust::detail::backend::dereference(first);
      ++out_false;
    } // end else
  }

  return thrust::make_pair(out_true, out_false);
}


} // end namespace detail
} // end namespace cpp
} // end namespace system
} // end namespace thrust

