/*
 *  Copyright 2008-2010 NVIDIA Corporation
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


/*! \file partition.inl
 *  \brief Inline file for partition.h.
 */

#include <thrust/iterator/iterator_traits.h>
#include <thrust/detail/dispatch/partition.h>

namespace thrust
{

template<typename ForwardIterator,
         typename Predicate>
  ForwardIterator partition(ForwardIterator first,
                            ForwardIterator last,
                            Predicate pred)
{
  return detail::dispatch::partition(first, last, pred,
    typename thrust::iterator_space<ForwardIterator>::type());
} // end partition()


template<typename ForwardIterator,
         typename Predicate>
  ForwardIterator stable_partition(ForwardIterator first,
                                   ForwardIterator last,
                                   Predicate pred)
{
  return detail::dispatch::stable_partition(first, last, pred,
    typename thrust::iterator_space<ForwardIterator>::type());
} // end stable_partition()


namespace experimental
{

template<typename ForwardIterator1,
         typename ForwardIterator2,
         typename Predicate>
  ForwardIterator2 partition_copy(ForwardIterator1 first,
                                  ForwardIterator1 last,
                                  ForwardIterator2 result,
                                  Predicate pred)
{
  return thrust::detail::dispatch::partition_copy(first, last, result, pred,
    typename thrust::iterator_space<ForwardIterator1>::type(),
    typename thrust::iterator_space<ForwardIterator2>::type());
} // end partition_copy()

template<typename ForwardIterator1,
         typename ForwardIterator2,
         typename Predicate>
  ForwardIterator2 stable_partition_copy(ForwardIterator1 first,
                                         ForwardIterator1 last,
                                         ForwardIterator2 result,
                                         Predicate pred)
{
  return thrust::detail::dispatch::stable_partition_copy(first, last, result, pred,
    typename thrust::iterator_space<ForwardIterator1>::type(),
    typename thrust::iterator_space<ForwardIterator2>::type());
} // end stable_partition_copy()

} // end namespace experimental

} // end thrust

