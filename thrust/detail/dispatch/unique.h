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
 *  \brief Dispatch layer for unique().
 */

#pragma once

#include <thrust/iterator/iterator_categories.h>

#include <algorithm>
#include <thrust/detail/device/unique.h>

namespace thrust
{

namespace detail
{

namespace dispatch
{

///////////////
// Host Path //
///////////////

template <typename ForwardIterator, typename BinaryPredicate>
ForwardIterator unique(ForwardIterator first, ForwardIterator last,
                       BinaryPredicate binary_pred,
                       thrust::host_space_tag)
{
    return std::unique(first, last, binary_pred);
}


/////////////////
// Device Path //
/////////////////

template <typename ForwardIterator, typename BinaryPredicate>
ForwardIterator unique(ForwardIterator first, ForwardIterator last,
                       BinaryPredicate binary_pred,
                       thrust::device_space_tag)
{
    return thrust::detail::device::unique(first, last, binary_pred);
}

} // end namespace dispatch

} // end namespace detail

} // end namespace thrust

