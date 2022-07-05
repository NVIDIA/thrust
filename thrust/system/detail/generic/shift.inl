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
#include <thrust/detail/temporary_array.h>
#include <thrust/system/detail/generic/shift.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/shift.h>
#include <thrust/copy.h>

THRUST_NAMESPACE_BEGIN
namespace system
{
namespace detail
{
namespace generic
{


template<typename DerivedPolicy,
         typename ForwardIterator>
__host__ __device__
  ForwardIterator shift_left(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                             ForwardIterator first,
                             ForwardIterator last,
                             typename thrust::iterator_traits<ForwardIterator>::difference_type n)
{
  if (n <= 0) {
    return last;
  }

  const auto len = thrust::distance(first, last);
  if (n >= len) {
    return first;
  }

  ForwardIterator new_first = first;
  thrust::advance(new_first, n)

  // copy partial input to temp buffer
  typedef typename thrust::iterator_traits<ForwardIterator>::value_type InputType;
  thrust::detail::temporary_array<InputType, DerivedPolicy> temp(exec, new_first, last);

  // copy all back to input
  return thrust::copy(exec, temp.begin(), temp.end(), first);
} // end shift_left()


template<typename DerivedPolicy,
         typename ForwardIterator>
__host__ __device__
  ForwardIterator shift_right(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                             ForwardIterator first,
                             ForwardIterator last,
                             typename thrust::iterator_traits<ForwardIterator>::difference_type n)
{
  if (n <= 0) {
    return first;
  }

  const auto len = thrust::distance(first, last);
  if (n >= len) {
    return last;
  }

  ForwardIterator new_last = first;
  thrust::advance(new_last, len - n)

  // copy full input to temp buffer
  typedef typename thrust::iterator_traits<ForwardIterator>::value_type InputType;
  thrust::detail::temporary_array<InputType, DerivedPolicy> temp(exec, first, new_last);

  // copy partially back to input
  thrust::advance(first, n)
  thrust::copy(exec, temp.begin(), temp.end(), first);
  return first;
} // end shift_right()

} // end generic
} // end detail
} // end system
THRUST_NAMESPACE_END
