/*
 *  Copyright 2008-2009 NVIDIA Corporation
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


/*! \file unique.h
 *  \brief Device implementation of unique()
 */

#pragma once

#include <thrust/detail/device/generic/unique.h>

namespace thrust
{
namespace detail
{
namespace device
{

template <typename ForwardIterator,
          typename BinaryPredicate>
ForwardIterator unique(ForwardIterator first,
                       ForwardIterator last,
                       BinaryPredicate binary_pred)
{
    return thrust::detail::device::generic::unique(first, last, binary_pred);
}

template <typename InputIterator,
          typename OutputIterator,
          typename BinaryPredicate>
OutputIterator unique_copy(InputIterator first,
                           InputIterator last,
                           OutputIterator output,
                           BinaryPredicate binary_pred)
{
    return thrust::detail::device::generic::unique_copy(first, last, output, binary_pred);
}

} // end namespace device
} // end namespace detail
} // end namespace thrust

