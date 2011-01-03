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
 *  \brief Dispatch layer for the partition functions.
 */

#pragma once

#include <thrust/iterator/iterator_traits.h>

#include <thrust/detail/host/partition.h>
#include <thrust/detail/device/partition.h>

namespace thrust
{

namespace detail
{

namespace dispatch
{

////////////////
// Host Paths //
////////////////
template<typename ForwardIterator,
         typename Predicate>
  ForwardIterator partition(ForwardIterator first,
                            ForwardIterator last,
                            Predicate pred,
                            thrust::host_space_tag)
{
    return thrust::detail::host::partition(first, last, pred);
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
                   Predicate pred,
                   thrust::host_space_tag,
                   thrust::host_space_tag,
                   thrust::host_space_tag)
{
    return thrust::detail::host::partition_copy(first, last, out_true, out_false, pred);
}

template<typename ForwardIterator,
         typename Predicate>
  ForwardIterator stable_partition(ForwardIterator first,
                                   ForwardIterator last,
                                   Predicate pred,
                                   thrust::host_space_tag)
{
    return thrust::detail::host::stable_partition(first, last, pred);
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
                          Predicate pred,
                          thrust::host_space_tag,
                          thrust::host_space_tag,
                          thrust::host_space_tag)
{
    return thrust::detail::host::stable_partition_copy(first, last, out_true, out_false, pred);
}

template<typename ForwardIterator1,
         typename ForwardIterator2,
         typename Predicate>
  ForwardIterator2 stable_partition_copy(ForwardIterator1 first,
                                         ForwardIterator1 last,
                                         ForwardIterator2 result,
                                         Predicate pred,
                                         thrust::host_space_tag,
                                         thrust::host_space_tag)
{
    return thrust::detail::host::stable_partition_copy(first, last, result, pred);
}


//////////////////
// Device Paths //
//////////////////
template<typename ForwardIterator,
         typename Predicate>
  ForwardIterator partition(ForwardIterator first,
                            ForwardIterator last,
                            Predicate pred,
                            thrust::device_space_tag)
{
    return thrust::detail::device::partition(first, last, pred);
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
                   Predicate pred,
                   thrust::device_space_tag,
                   thrust::device_space_tag,
                   thrust::device_space_tag)
{
    return thrust::detail::device::partition_copy(first, last, out_true, out_false, pred);
}

template<typename ForwardIterator,
         typename Predicate>
  ForwardIterator stable_partition(ForwardIterator first,
                                   ForwardIterator last,
                                   Predicate pred,
                                   thrust::device_space_tag)
{
    return thrust::detail::device::stable_partition(first, last, pred);
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
                   Predicate pred,
                   thrust::device_space_tag,
                   thrust::device_space_tag,
                   thrust::device_space_tag)
{
    return thrust::detail::device::stable_partition_copy(first, last, out_true, out_false, pred);
}

} // end namespace dispatch

} // end namespace detail

} // end namespace thrust

