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
 *  \brief Device interface to partition functions.
 */

#pragma once

#include <thrust/iterator/detail/minimum_space.h>
#include <thrust/iterator/iterator_traits.h>

#include <thrust/detail/backend/generic/partition.h>
#include <thrust/detail/backend/cpp/partition.h>
#include <thrust/pair.h>

namespace thrust
{
namespace detail
{
namespace backend
{
namespace dispatch
{



template<typename ForwardIterator,
         typename Predicate,
         typename Backend>
  ForwardIterator stable_partition(ForwardIterator first,
                                   ForwardIterator last,
                                   Predicate pred,
                                   Backend)
{
    return thrust::detail::backend::generic::stable_partition(first, last, pred);
}

template<typename ForwardIterator,
         typename Predicate>
  ForwardIterator stable_partition(ForwardIterator first,
                                   ForwardIterator last,
                                   Predicate pred,
                                   thrust::host_space_tag)
{
    return thrust::detail::backend::cpp::stable_partition(first, last, pred);
}



template<typename InputIterator,
         typename OutputIterator1,
         typename OutputIterator2,
         typename Predicate,
         typename Backend>
  thrust::pair<OutputIterator1,OutputIterator2>
    stable_partition_copy(InputIterator first,
                          InputIterator last,
                          OutputIterator1 out_true,
                          OutputIterator2 out_false,
                          Predicate pred,
                          Backend)
{
    return thrust::detail::backend::generic::stable_partition_copy(first, last, out_true, out_false, pred);
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
                          thrust::host_space_tag)
{
    return thrust::detail::backend::cpp::stable_partition_copy(first, last, out_true, out_false, pred);
}



template<typename ForwardIterator,
         typename Predicate,
         typename Backend>
  ForwardIterator partition(ForwardIterator first,
                            ForwardIterator last,
                            Predicate pred,
                            Backend)
{
    return thrust::detail::backend::generic::partition(first, last, pred);
}

template<typename ForwardIterator,
         typename Predicate>
  ForwardIterator partition(ForwardIterator first,
                            ForwardIterator last,
                            Predicate pred,
                            thrust::host_space_tag)
{
    return thrust::detail::backend::cpp::partition(first, last, pred);
}



template<typename InputIterator,
         typename OutputIterator1,
         typename OutputIterator2,
         typename Predicate,
         typename Backend>
  thrust::pair<OutputIterator1,OutputIterator2>
    partition_copy(InputIterator first,
                   InputIterator last,
                   OutputIterator1 out_true,
                   OutputIterator2 out_false,
                   Predicate pred,
                   Backend)
{
    return thrust::detail::backend::generic::partition_copy(first, last, out_true, out_false, pred);
}

} // end namespace dispatch



template<typename ForwardIterator,
         typename Predicate>
  ForwardIterator stable_partition(ForwardIterator first,
                                   ForwardIterator last,
                                   Predicate pred)
{
    return thrust::detail::backend::dispatch::stable_partition(first, last, pred,
        typename thrust::iterator_space<ForwardIterator>::type());
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
    return thrust::detail::backend::dispatch::stable_partition_copy(first, last, out_true, out_false, pred,
        typename thrust::detail::minimum_space<
          typename thrust::iterator_space<InputIterator>::type,
          typename thrust::iterator_space<OutputIterator1>::type,
          typename thrust::iterator_space<OutputIterator2>::type
        >::type());
}

template<typename ForwardIterator,
         typename Predicate>
  ForwardIterator partition(ForwardIterator first,
                            ForwardIterator last,
                            Predicate pred)
{
    return thrust::detail::backend::dispatch::partition(first, last, pred,
        typename thrust::iterator_space<ForwardIterator>::type());
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
    return thrust::detail::backend::dispatch::partition_copy(first, last, out_true, out_false, pred,
        typename thrust::detail::minimum_space<
          typename thrust::iterator_space<InputIterator>::type,
          typename thrust::iterator_space<OutputIterator1>::type,
          typename thrust::iterator_space<OutputIterator2>::type
        >::type());
}

} // end namespace backend
} // end namespace detail
} // end namespace thrust

