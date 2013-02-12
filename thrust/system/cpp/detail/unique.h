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
#include <thrust/pair.h>
#include <thrust/system/detail/internal/scalar/unique.h>

namespace thrust
{
namespace system
{
namespace cpp
{
namespace detail
{

template<typename DerivedPolicy,
         typename ForwardIterator,
         typename BinaryPredicate>
  ForwardIterator unique(execution_policy<DerivedPolicy> &,
                         ForwardIterator first,
                         ForwardIterator last,
                         BinaryPredicate binary_pred)
{
  return thrust::system::detail::internal::scalar::unique(first, last, binary_pred);
}

template<typename DerivedPolicy,
         typename InputIterator,
         typename OutputIterator,
         typename BinaryPredicate>
  OutputIterator unique_copy(execution_policy<DerivedPolicy> &,
                             InputIterator first,
                             InputIterator last,
                             OutputIterator output,
                             BinaryPredicate binary_pred)
{
  return thrust::system::detail::internal::scalar::unique_copy(first, last, output, binary_pred);
}

} // end namespace detail
} // end namespace cpp 
} // end namespace system
} // end namespace thrust

