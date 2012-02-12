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


/*! \file partition.inl
 *  \brief Inline file for partition.h.
 */

#include <thrust/detail/config.h>
#include <thrust/partition.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/system/detail/generic/select_system.h>
#include <thrust/system/detail/generic/partition.h>
#include <thrust/detail/adl_helper.h>

namespace thrust
{

template<typename ForwardIterator,
         typename Predicate>
  ForwardIterator partition(ForwardIterator first,
                            ForwardIterator last,
                            Predicate pred)
{
  using thrust::system::detail::generic::select_system;
  using thrust::system::detail::generic::partition;

  typedef typename thrust::iterator_system<ForwardIterator>::type system;

  return partition(select_system(system()), first, last, pred);
} // end partition()


template<typename ForwardIterator,
         typename Predicate>
  ForwardIterator stable_partition(ForwardIterator first,
                                   ForwardIterator last,
                                   Predicate pred)
{
  using thrust::system::detail::generic::select_system;
  using thrust::system::detail::generic::stable_partition;

  typedef typename thrust::iterator_system<ForwardIterator>::type system;

  return stable_partition(select_system(system()), first, last, pred);
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
  using thrust::system::detail::generic::select_system;
  using thrust::system::detail::generic::partition_copy;

  typedef typename thrust::iterator_system<InputIterator>::type   system1;
  typedef typename thrust::iterator_system<OutputIterator1>::type system2;
  typedef typename thrust::iterator_system<OutputIterator2>::type system3;

  return partition_copy(select_system(system1(),system2(),system3()), first, last, out_true, out_false, pred);
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
  using thrust::system::detail::generic::select_system;
  using thrust::system::detail::generic::stable_partition_copy;

  typedef typename thrust::iterator_system<InputIterator>::type   system1;
  typedef typename thrust::iterator_system<OutputIterator1>::type system2;
  typedef typename thrust::iterator_system<OutputIterator2>::type system3;

  return stable_partition_copy(select_system(system1(),system2(),system3()), first, last, out_true, out_false, pred);
} // end stable_partition_copy()


template<typename ForwardIterator, typename Predicate>
  ForwardIterator partition_point(ForwardIterator first,
                                  ForwardIterator last,
                                  Predicate pred)
{
  using thrust::system::detail::generic::select_system;
  using thrust::system::detail::generic::partition_point;

  typedef typename thrust::iterator_system<ForwardIterator>::type system;

  return partition_point(select_system(system()), first, last, pred);
} // end partition_point()


template<typename InputIterator, typename Predicate>
  bool is_partitioned(InputIterator first,
                      InputIterator last,
                      Predicate pred)
{
  using thrust::system::detail::generic::select_system;
  using thrust::system::detail::generic::is_partitioned;

  typedef typename thrust::iterator_system<InputIterator>::type system;

  return is_partitioned(select_system(system()), first, last, pred);
} // end is_partitioned()


} // end thrust

