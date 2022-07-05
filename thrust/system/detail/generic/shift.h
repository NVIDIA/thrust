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
#include <thrust/system/detail/generic/tag.h>
#include <thrust/iterator/iterator_traits.h>

THRUST_NAMESPACE_BEGIN
namespace system
{
namespace detail
{
namespace generic
{


template <typename DerivedPolicy, typename ForwardIterator>
__host__ __device__
  ForwardIterator shift_left(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                             ForwardIterator first,
                             ForwardIterator last,
                             typename thrust::iterator_traits<ForwardIterator>::difference_type n);


template <typename DerivedPolicy, typename ForwardIterator>
__host__ __device__
  ForwardIterator shift_right(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                             ForwardIterator first,
                             ForwardIterator last,
                             typename thrust::iterator_traits<ForwardIterator>::difference_type n);

} // end generic
} // end detail
} // end system
THRUST_NAMESPACE_END

#include <thrust/system/detail/generic/shift.inl>

