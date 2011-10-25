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

#include <thrust/detail/config.h>
#include <thrust/partition.h>
#include <thrust/detail/backend/generic/select_system.h>
#include <thrust/detail/backend/generic/partition.h>
#include <thrust/iterator/iterator_traits.h>

// XXX make the backend-specific versions available
// XXX try to eliminate the need for these
#include <thrust/system/cpp/detail/partition.h>

namespace thrust
{

template<typename ForwardIterator,
         typename Predicate>
  ForwardIterator partition(ForwardIterator first,
                            ForwardIterator last,
                            Predicate pred)
{
  using thrust::detail::backend::generic::select_system;
  using thrust::detail::backend::generic::partition;

  typedef typename thrust::iterator_space<ForwardIterator>::type space;

  return partition(select_system(space()), first, last, pred);
} // end partition()


template<typename ForwardIterator,
         typename Predicate>
  ForwardIterator stable_partition(ForwardIterator first,
                                   ForwardIterator last,
                                   Predicate pred)
{
  using thrust::detail::backend::generic::select_system;
  using thrust::detail::backend::generic::stable_partition;

  typedef typename thrust::iterator_space<ForwardIterator>::type space;

  return stable_partition(select_system(space()), first, last, pred);
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
  using thrust::detail::backend::generic::select_system;
  using thrust::detail::backend::generic::partition_copy;

  typedef typename thrust::iterator_space<InputIterator>::type   space1;
  typedef typename thrust::iterator_space<OutputIterator1>::type space2;
  typedef typename thrust::iterator_space<OutputIterator2>::type space3;

  return partition_copy(select_system(space1(),space2(),space3()), first, last, out_true, out_false, pred);
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
  using thrust::detail::backend::generic::select_system;
  using thrust::detail::backend::generic::stable_partition_copy;

  typedef typename thrust::iterator_space<InputIterator>::type   space1;
  typedef typename thrust::iterator_space<OutputIterator1>::type space2;
  typedef typename thrust::iterator_space<OutputIterator2>::type space3;

  return stable_partition_copy(select_system(space1(),space2(),space3()), first, last, out_true, out_false, pred);
} // end stable_partition_copy()


template<typename ForwardIterator, typename Predicate>
  ForwardIterator partition_point(ForwardIterator first,
                                  ForwardIterator last,
                                  Predicate pred)
{
  using thrust::detail::backend::generic::select_system;
  using thrust::detail::backend::generic::partition_point;

  typedef typename thrust::iterator_space<ForwardIterator>::type space;

  return partition_point(select_system(space()), first, last, pred);
} // end partition_point()


template<typename InputIterator, typename Predicate>
  bool is_partitioned(InputIterator first,
                      InputIterator last,
                      Predicate pred)
{
  using thrust::detail::backend::generic::select_system;
  using thrust::detail::backend::generic::is_partitioned;

  typedef typename thrust::iterator_space<InputIterator>::type space;

  return is_partitioned(select_system(space()), first, last, pred);
} // end is_partitioned()


} // end thrust

