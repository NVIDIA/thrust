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
#include <thrust/system/detail/generic/tag.h>

namespace thrust
{
namespace system
{
namespace detail
{
namespace generic
{


template<typename System,
         typename ForwardIterator>
ForwardIterator unique(thrust::dispatchable<System> &system,
                       ForwardIterator first,
                       ForwardIterator last);


template<typename System,
         typename ForwardIterator,
         typename BinaryPredicate>
ForwardIterator unique(thrust::dispatchable<System> &system,
                       ForwardIterator first,
                       ForwardIterator last,
                       BinaryPredicate binary_pred);


template<typename System,
         typename InputIterator,
         typename OutputIterator>
OutputIterator unique_copy(thrust::dispatchable<System> &system,
                           InputIterator first,
                           InputIterator last,
                           OutputIterator output);


template<typename System,
         typename InputIterator,
         typename OutputIterator,
         typename BinaryPredicate>
OutputIterator unique_copy(thrust::dispatchable<System> &system,
                           InputIterator first,
                           InputIterator last,
                           OutputIterator output,
                           BinaryPredicate binary_pred);


} // end namespace generic
} // end namespace detail
} // end namespace system
} // end namespace thrust

#include <thrust/system/detail/generic/unique.inl>

