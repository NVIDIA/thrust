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


/*! \file mismatch.h
 *  \brief Dispatch layer of the mismatch function.
 */

#pragma once

#include <thrust/iterator/iterator_categories.h>

#include <thrust/detail/host/mismatch.h>
#include <thrust/detail/device/mismatch.h>

namespace thrust
{
namespace detail
{
namespace dispatch
{

////////////////
// Host Paths //
////////////////
template <typename InputIterator1, typename InputIterator2, typename BinaryPredicate>
thrust::pair<InputIterator1, InputIterator2> mismatch(InputIterator1 first1,
                                                      InputIterator1 last1,
                                                      InputIterator2 first2,
                                                      BinaryPredicate pred,
                                                      thrust::host_space_tag,
                                                      thrust::host_space_tag)
{
    return thrust::detail::host::mismatch(first1, last1, first2, pred);
}


//////////////////
// Device Paths //
//////////////////
template <typename InputIterator1, typename InputIterator2, typename BinaryPredicate>
thrust::pair<InputIterator1, InputIterator2> mismatch(InputIterator1 first1,
                                                      InputIterator1 last1,
                                                      InputIterator2 first2,
                                                      BinaryPredicate pred,
                                                      thrust::device_space_tag,
                                                      thrust::device_space_tag)
{
    return thrust::detail::device::mismatch(first1, last1, first2, pred);
}

///////////////
// Any Paths //
///////////////
template <typename InputIterator1, typename InputIterator2, typename BinaryPredicate>
thrust::pair<InputIterator1, InputIterator2> mismatch(InputIterator1 first1,
                                                      InputIterator1 last1,
                                                      InputIterator2 first2,
                                                      BinaryPredicate pred,
                                                      thrust::any_space_tag,
                                                      thrust::any_space_tag)
{
    // default to device
    return thrust::detail::device::mismatch(first1, last1, first2, pred);
}

} // end namespace dispatch
} // end namespace detail
} // end namespace thrust

