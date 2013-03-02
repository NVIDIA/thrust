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
#include <thrust/system/detail/sequential/execution_policy.h>

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
  ForwardIterator partition(ForwardIterator first,
                            ForwardIterator last,
                            Predicate pred)
{
  return partition(thrust::system::detail::sequential::seq, first, last, pred);
}

template<typename ForwardIterator,
         typename Predicate>
  ForwardIterator stable_partition(ForwardIterator first,
                                   ForwardIterator last,
                                   Predicate pred)
{
  return stable_partition(thrust::system::detail::sequential::seq, first, last, pred);
}

template<typename ForwardIterator,
         typename InputIterator,
         typename Predicate>
  ForwardIterator stable_partition(ForwardIterator first,
                                   ForwardIterator last,
                                   InputIterator stencil,
                                   Predicate pred)
{
  return stable_partition(thrust::system::detail::sequential::seq, first, last, stencil, pred);
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
  return stable_partition_copy(thrust::system::detail::sequential::seq, first, last, out_true, out_false, pred);
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
  return stable_partition_copy(thrust::system::detail::sequential::seq, first, last, stencil, out_true, out_false, pred);
}

} // end namespace scalar
} // end namespace internal
} // end namespace detail
} // end namespace system
} // end namespace thrust

