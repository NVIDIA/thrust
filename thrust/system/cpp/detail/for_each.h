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

#pragma once

#include <thrust/detail/config.h>
#include <thrust/system/cpp/detail/execution_policy.h>
#include <thrust/system/detail/internal/scalar/for_each.h>

namespace thrust
{
namespace system
{
namespace cpp
{
namespace detail
{


template<typename DerivedPolicy,
         typename InputIterator,
         typename UnaryFunction>
InputIterator for_each(thrust::system::cpp::detail::execution_policy<DerivedPolicy> &,
                       InputIterator first,
                       InputIterator last,
                       UnaryFunction f)
{
  return thrust::system::detail::internal::scalar::for_each(first, last, f);
}

template<typename DerivedPolicy,
         typename InputIterator,
         typename Size,
         typename UnaryFunction>
InputIterator for_each_n(thrust::system::cpp::detail::execution_policy<DerivedPolicy> &,
                         InputIterator first,
                         Size n,
                         UnaryFunction f)
{
  return thrust::system::detail::internal::scalar::for_each_n(first, n, f);
}

} // end namespace detail
} // end namespace cpp
} // end namespace system
} // end namespace thrust

