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
 *  \brief cpp implementations of partition functions
 */

#pragma once

#include <thrust/pair.h>
#include <thrust/system/cpp/detail/execution_policy.h>
#include <thrust/system/detail/internal/scalar/partition.h>

namespace thrust
{
namespace system
{
namespace cpp
{
namespace detail
{

template<typename ForwardIterator,
         typename Predicate>
  ForwardIterator stable_partition(tag,
                                   ForwardIterator first,
                                   ForwardIterator last,
                                   Predicate pred)
{
  return thrust::system::detail::internal::scalar::stable_partition(first, last, pred);
}

template<typename ForwardIterator,
         typename InputIterator,
         typename Predicate>
  ForwardIterator stable_partition(tag,
                                   ForwardIterator first,
                                   ForwardIterator last,
                                   InputIterator stencil,
                                   Predicate pred)
{
  return thrust::system::detail::internal::scalar::stable_partition(first, last, stencil, pred);
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
  return thrust::system::detail::internal::scalar::stable_partition_copy(first, last, out_true, out_false, pred);
}

template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator1,
         typename OutputIterator2,
         typename Predicate>
  thrust::pair<OutputIterator1,OutputIterator2>
    stable_partition_copy(tag,
                          InputIterator1 first,
                          InputIterator1 last,
                          InputIterator2 stencil,
                          OutputIterator1 out_true,
                          OutputIterator2 out_false,
                          Predicate pred)
{
  return thrust::system::detail::internal::scalar::stable_partition_copy(first, last, stencil, out_true, out_false, pred);
}

} // end namespace detail
} // end namespace cpp
} // end namespace system
} // end namespace thrust

