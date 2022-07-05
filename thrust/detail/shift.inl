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
#include <thrust/shift.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/system/detail/generic/select_system.h>
#include <thrust/system/detail/generic/shift.h>
#include <thrust/system/detail/adl/shift.h>

THRUST_NAMESPACE_BEGIN

__thrust_exec_check_disable__
template<typename DerivedPolicy, typename ForwardIterator>
__host__ __device__
  ForwardIterator shift_left(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
               ForwardIterator first,
               ForwardIterator last,
               typename thrust::iterator_traits<ForwardIterator>::difference_type n)
{
  using thrust::system::detail::generic::shift_left;
  return shift_left(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, last, n);
} // end shift_left()


__thrust_exec_check_disable__
template<typename DerivedPolicy, typename ForwardIterator>
__host__ __device__
  ForwardIterator shift_right(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
               ForwardIterator first,
               ForwardIterator last,
               typename thrust::iterator_traits<ForwardIterator>::difference_type n)
{
  using thrust::system::detail::generic::shift_left;
  return shift_right(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, last, n);
} // end shift_right()


template<typename ForwardIterator>
__host__ __device__
  ForwardIterator shift_left(ForwardIterator first,
                             ForwardIterator last,
                             typename thrust::iterator_traits<ForwardIterator>::difference_type n)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<ForwardIterator>::type System;

  System system;

  return thrust::shift_left(select_system(system), first, last, n);
} // end shift_left()


template<typename ForwardIterator>
__host__ __device__
  ForwardIterator shift_right(ForwardIterator first,
                             ForwardIterator last,
                             typename thrust::iterator_traits<ForwardIterator>::difference_type n)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<ForwardIterator>::type System;

  System system;

  return thrust::shift_right(select_system(system), first, last, n);
} // end shift_right()

THRUST_NAMESPACE_END
