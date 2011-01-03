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
 *  \brief Host implementations of partition functions
 */

#pragma once

#include <thrust/pair.h>
#include <thrust/host_vector.h>

namespace thrust
{
namespace detail
{
namespace host
{

template <typename ForwardIterator1,
          typename ForwardIterator2>
void iter_swap(ForwardIterator1 iter1, ForwardIterator2 iter2)
{
    typedef typename thrust::iterator_value<ForwardIterator1>::type T;

    T temp = *iter1;
    *iter1 = *iter2;
    *iter2 =  temp;
}

template<typename ForwardIterator,
         typename Predicate>
  ForwardIterator partition(ForwardIterator first,
                            ForwardIterator last,
                            Predicate pred)
{
    if (first == last)
        return first;

    while (pred(*first))
    {
        if (++first == last)
            return first;
    }

    ForwardIterator next = first;

    while (++next != last)
    {
        if (pred(*next))
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
    typedef typename thrust::iterator_value<ForwardIterator>::type T;

    typedef thrust::host_vector<T>        TempRange;
    typedef typename TempRange::iterator  TempIterator;

    TempRange temp(first, last);

    for(TempIterator iter = temp.begin(); iter != temp.end(); ++iter)
    {
        if (pred(*iter))
        {
            *first = *iter;
            ++first;
        }
    }

    ForwardIterator middle = first;

    for(TempIterator iter = temp.begin(); iter != temp.end(); ++iter)
    {
        if (!bool(pred(*iter)))
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
  for(; first != last; ++first)
  {
    if(pred(*first))
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


template<typename InputIterator,
         typename OutputIterator1,
         typename OutputIterator2,
         typename Predicate>
  thrust::pair<OutputIterator1,OutputIterator2>
    partition_copy(InputIterator first,
                   InputIterator last,
                   OutputIterator1 out_true,
                   OutputIterator2 out_false,
                   Predicate pred)
{
  return thrust::detail::host::stable_partition_copy(first,last,out_true,out_false,pred);
}


} // end namespace host
} // end namespace detail
} // end namespace thrust

