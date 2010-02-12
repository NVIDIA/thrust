/*
 *  Copyright 2008-2010 NVIDIA Corporation
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


/*! \file scan.h
 *  \brief Dispatch layer for the scan functions.
 */

#pragma once

#include <thrust/iterator/iterator_traits.h>

#include <thrust/detail/host/scan.h>
#include <thrust/detail/device/scan.h>

namespace thrust
{

namespace detail
{

namespace dispatch
{

////////////////
// Host Paths //
////////////////

template<typename InputIterator,
         typename OutputIterator,
         typename AssociativeOperator>
  OutputIterator inclusive_scan(InputIterator first,
                                InputIterator last,
                                OutputIterator result,
                                AssociativeOperator binary_op,
                                thrust::host_space_tag,
                                thrust::host_space_tag)
{
    return thrust::detail::host::inclusive_scan(first, last, result, binary_op);
}


template<typename InputIterator,
         typename OutputIterator,
         typename T,
         typename AssociativeOperator>
  OutputIterator exclusive_scan(InputIterator first,
                                InputIterator last,
                                OutputIterator result,
                                T init,
                                AssociativeOperator binary_op,
                                thrust::host_space_tag,
                                thrust::host_space_tag)
{
    return thrust::detail::host::exclusive_scan(first, last, result, init, binary_op);
}


//////////////////
// Device Paths //
//////////////////

template<typename InputIterator,
         typename OutputIterator,
         typename AssociativeOperator>
  OutputIterator inclusive_scan(InputIterator first,
                                InputIterator last,
                                OutputIterator result,
                                AssociativeOperator binary_op,
                                thrust::device_space_tag,
                                thrust::device_space_tag)
{
    return thrust::detail::device::inclusive_scan(first, last, result, binary_op);
}


template<typename InputIterator,
         typename OutputIterator,
         typename T,
         typename AssociativeOperator>
  OutputIterator exclusive_scan(InputIterator first,
                                InputIterator last,
                                OutputIterator result,
                                T init,
                                AssociativeOperator binary_op,
                                thrust::device_space_tag,
                                thrust::device_space_tag)
{
    return thrust::detail::device::exclusive_scan(first, last, result, init, binary_op);
}

} // end namespace dispatch

} // end namespace detail

} // end namespace thrust

