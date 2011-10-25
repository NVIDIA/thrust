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


/*! \file reduce.h
 *  \brief OpenMP implementation of reduce algorithms.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/backend/omp/partition.h>
#include <thrust/detail/backend/generic/partition.h>

namespace thrust
{
namespace detail
{
namespace backend
{
namespace omp
{


template<typename ForwardIterator,
         typename Predicate>
  ForwardIterator partition(tag,
                            ForwardIterator first,
                            ForwardIterator last,
                            Predicate pred)
{
  // omp prefers generic::partition to cpp::partition
  return thrust::detail::backend::generic::partition(tag(), first, last, pred);
} // end partition()


template<typename ForwardIterator,
         typename Predicate>
  ForwardIterator stable_partition(tag,
                                   ForwardIterator first,
                                   ForwardIterator last,
                                   Predicate pred)
{
  // omp prefers generic::stable_partition to cpp::partition
  return thrust::detail::backend::generic::partition(tag(), first, last, pred);
} // end stable_partition()


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
  // omp prefers generic::stable_partition_copy to cpp::stable_partition_copy
  return thrust::detail::backend::generic::stable_partition_copy(tag(), first, last, out_true, out_false, pred);
} // end stable_partition_copy()


} // end namespace omp
} // end namespace backend
} // end namespace detail
} // end namespace thrust

#include <thrust/detail/backend/omp/partition.inl>

