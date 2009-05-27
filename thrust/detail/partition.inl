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


/*! \file partition.inl
 *  \brief Inline file for partition.h.
 */

#include <thrust/partition.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/detail/dispatch/partition.h>

namespace thrust
{

namespace experimental
{

template<typename ForwardIterator,
         typename OutputIterator,
         typename Predicate>
  OutputIterator stable_partition_copy(ForwardIterator begin,
                                       ForwardIterator end,
                                       OutputIterator result,
                                       Predicate pred)
{
  return thrust::detail::dispatch::stable_partition_copy(begin, end, result, pred,
    typename thrust::iterator_traits<ForwardIterator>::iterator_category(),
    typename thrust::iterator_traits<OutputIterator>::iterator_category());
} // end stable_partition_copy()


template<typename ForwardIterator,
         typename OutputIterator,
         typename Predicate>
  OutputIterator partition_copy(ForwardIterator begin,
                                ForwardIterator end,
                                OutputIterator result,
                                Predicate pred)
{
  return thrust::experimental::stable_partition_copy(begin,end,result,pred);
} // end partition_copy()

} // end namespace experimental


template<typename ForwardIterator,
         typename Predicate>
  ForwardIterator stable_partition(ForwardIterator begin,
                                   ForwardIterator end,
                                   Predicate pred)
{
  return detail::dispatch::stable_partition(begin, end, pred,
    typename thrust::iterator_traits<ForwardIterator>::iterator_category());
} // end stable_partition_copy()


template<typename ForwardIterator,
         typename Predicate>
  ForwardIterator partition(ForwardIterator begin,
                            ForwardIterator end,
                            Predicate pred)
{
  return thrust::stable_partition(begin,end,pred);
} // end partition()

} // end thrust

