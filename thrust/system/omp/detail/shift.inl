/*
 *  Copyright 2022 NVIDIA Corporation
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

#pragma once

#include <thrust/detail/config.h>
#include <thrust/system/omp/detail/shift.h>
#include <thrust/system/detail/generic/shift.h>
#include <thrust/system/detail/sequential/shift.h>
#include <thrust/iterator/iterator_traits.h>


THRUST_NAMESPACE_BEGIN
namespace system
{
namespace omp
{
namespace detail
{
namespace dispatch
{


template<typename DerivedPolicy,
         typename ForwardIterator>
ForwardIterator shift_left(execution_policy<DerivedPolicy> &exec,
                    ForwardIterator first,
                    ForwardIterator last,
                    typename thrust::iterator_traits<ForwardIterator>::difference_type n,
                    thrust::incrementable_traversal_tag)
{
  return thrust::system::detail::sequential::shift_left(exec, first, last, n);
} // end shift_left()


template<typename DerivedPolicy,
         typename ForwardIterator>
ForwardIterator shift_left(execution_policy<DerivedPolicy> &exec,
                    ForwardIterator first,
                    ForwardIterator last,
                    typename thrust::iterator_traits<ForwardIterator>::difference_type n,
                    thrust::random_access_traversal_tag)
{
  return thrust::system::detail::generic::shift_left(exec, first, last, n);
} // end shift_left()


template<typename DerivedPolicy,
         typename ForwardIterator>
ForwardIterator shift_right(execution_policy<DerivedPolicy> &exec,
                    ForwardIterator first,
                    ForwardIterator last,
                    typename thrust::iterator_traits<ForwardIterator>::difference_type n,
                    thrust::incrementable_traversal_tag)
{
  return thrust::system::detail::sequential::shift_right(exec, first, last, n);
} // end shift_right()


template<typename DerivedPolicy,
         typename ForwardIterator>
ForwardIterator shift_right(execution_policy<DerivedPolicy> &exec,
                    ForwardIterator first,
                    ForwardIterator last,
                    typename thrust::iterator_traits<ForwardIterator>::difference_type n,
                    thrust::random_access_traversal_tag)
{
  return thrust::system::detail::generic::shift_right(exec, first, last, n);
} // end shift_right()


} // end dispatch


template<typename DerivedPolicy,
         typename ForwardIterator>
ForwardIterator shift_left(execution_policy<DerivedPolicy> &exec,
                    ForwardIterator first,
                    ForwardIterator last,
                    typename thrust::iterator_traits<ForwardIterator>::difference_type n)
{
  typedef typename thrust::iterator_traversal<ForwardIterator>::type traversal;

  return thrust::system::omp::detail::dispatch::shift_left(exec, first, last, n, traversal());
} // end shift_left()


template<typename DerivedPolicy,
         typename ForwardIterator>
ForwardIterator shift_right(execution_policy<DerivedPolicy> &exec,
                    ForwardIterator first,
                    ForwardIterator last,
                    typename thrust::iterator_traits<ForwardIterator>::difference_type n)
{
  typedef typename thrust::iterator_traversal<ForwardIterator>::type traversal;

  return thrust::system::omp::detail::dispatch::shift_right(exec, first, last, n, traversal());
} // end shift_right()


} // end namespace detail
} // end namespace omp
} // end namespace system
THRUST_NAMESPACE_END

