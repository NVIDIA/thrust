/*
 *  Copyright 2008-2011 NVIDIA Corporation
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


/*! \file find.h
 *  \brief Dispatch layer of the find functions.
 */

#pragma once

#include <thrust/iterator/iterator_categories.h>

#include <thrust/detail/host/find.h>
#include <thrust/detail/device/find.h>

namespace thrust
{
namespace detail
{
namespace dispatch
{

////////////////
// Host Paths //
////////////////
template <typename InputIterator, typename Predicate>
InputIterator find_if(InputIterator first,
                      InputIterator last,
                      Predicate pred,
                      thrust::host_space_tag)
{
    return thrust::detail::host::find_if(first, last, pred);
}

//////////////////
// Device Paths //
//////////////////
template <typename InputIterator, typename Predicate>
InputIterator find_if(InputIterator first,
                      InputIterator last,
                      Predicate pred,
                      thrust::device_space_tag)
{
    return thrust::detail::device::find_if(first, last, pred);
}

///////////////
// Any Paths //
///////////////
template <typename InputIterator, typename Predicate>
InputIterator find_if(InputIterator first,
                      InputIterator last,
                      Predicate pred,
                      thrust::any_space_tag)
{
    // default to device
    return thrust::detail::device::find_if(first, last, pred);
}

} // end namespace dispatch
} // end namespace detail
} // end namespace thrust

