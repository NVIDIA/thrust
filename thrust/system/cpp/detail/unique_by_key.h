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
#include <thrust/system/cpp/detail/tag.h>
#include <thrust/pair.h>
#include <thrust/system/detail/internal/scalar/unique_by_key.h>

namespace thrust
{
namespace system
{
namespace cpp
{
namespace detail
{

template<typename System,
         typename ForwardIterator1,
         typename ForwardIterator2,
         typename BinaryPredicate>
  thrust::pair<ForwardIterator1,ForwardIterator2>
    unique_by_key(dispatchable<System> &,
                  ForwardIterator1 keys_first, 
                  ForwardIterator1 keys_last,
                  ForwardIterator2 values_first,
                  BinaryPredicate binary_pred)
{
  return thrust::system::detail::internal::scalar::unique_by_key(keys_first, keys_last, values_first, binary_pred);
}


template<typename System,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator1,
         typename OutputIterator2,
         typename BinaryPredicate>
  thrust::pair<OutputIterator1,OutputIterator2>
    unique_by_key_copy(dispatchable<System> &,
                       InputIterator1 keys_first, 
                       InputIterator1 keys_last,
                       InputIterator2 values_first,
                       OutputIterator1 keys_output,
                       OutputIterator2 values_output,
                       BinaryPredicate binary_pred)
{
  return thrust::system::detail::internal::scalar::unique_by_key_copy(keys_first, keys_last, values_first, keys_output, values_output, binary_pred);
}

} // end namespace detail
} // end namespace cpp 
} // end namespace system
} // end namespace thrust

