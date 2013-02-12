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
#include <thrust/system/detail/internal/scalar/scan_by_key.h>

namespace thrust
{
namespace system
{
namespace cpp
{
namespace detail
{

template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename BinaryPredicate,
         typename BinaryFunction>
  OutputIterator inclusive_scan_by_key(tag,
                                       InputIterator1 first1,
                                       InputIterator1 last1,
                                       InputIterator2 first2,
                                       OutputIterator result,
                                       BinaryPredicate binary_pred,
                                       BinaryFunction binary_op)
{
  return thrust::system::detail::internal::scalar::inclusive_scan_by_key(first1, last1, first2, result, binary_pred, binary_op);
}


template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename T,
         typename BinaryPredicate,
         typename BinaryFunction>
  OutputIterator exclusive_scan_by_key(tag,
                                       InputIterator1 first1,
                                       InputIterator1 last1,
                                       InputIterator2 first2,
                                       OutputIterator result,
                                       T init,
                                       BinaryPredicate binary_pred,
                                       BinaryFunction binary_op)
{
  return thrust::system::detail::internal::scalar::exclusive_scan_by_key(first1, last1, first2, result, init, binary_pred, binary_op);
}

} // end namespace detail
} // end namespace cpp
} // end namespace system
} // end namespace thrust

