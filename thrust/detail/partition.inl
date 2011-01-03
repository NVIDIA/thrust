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


/*! \file partition.inl
 *  \brief Inline file for partition.h.
 */

#include <thrust/iterator/iterator_traits.h>
#include <thrust/detail/dispatch/partition.h>
#include <thrust/find.h>
#include <thrust/sort.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/detail/internal_functional.h>

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
  return thrust::detail::dispatch::partition_copy(first, last, out_true, out_false, pred,
    typename thrust::iterator_space<InputIterator>::type(),
    typename thrust::iterator_space<OutputIterator1>::type(),
    typename thrust::iterator_space<OutputIterator2>::type());
} // end partition_copy()


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
  return thrust::detail::dispatch::stable_partition_copy(first, last, out_true, out_false, pred,
    typename thrust::iterator_space<InputIterator>::type(),
    typename thrust::iterator_space<OutputIterator1>::type(),
    typename thrust::iterator_space<OutputIterator2>::type());
} // end stable_partition_copy()


template<typename ForwardIterator, typename Predicate>
  ForwardIterator partition_point(ForwardIterator first,
                                  ForwardIterator last,
                                  Predicate pred)
{
  return thrust::find_if_not(first, last, pred);
} // end partition_point()


template<typename InputIterator, typename Predicate>
  bool is_partitioned(InputIterator first,
                      InputIterator last,
                      Predicate pred)
{
  return thrust::is_sorted(thrust::make_transform_iterator(first, thrust::detail::not1(pred)),
                           thrust::make_transform_iterator(last,  thrust::detail::not1(pred)));
} // end is_partitioned()


} // end thrust

