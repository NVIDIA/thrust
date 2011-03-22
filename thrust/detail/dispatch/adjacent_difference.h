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


/*! \file adjacent_difference.h
 *  \brief Dispatch layer of the adjacent_difference function.
 */

#pragma once

#include <thrust/iterator/iterator_traits.h>

#include <thrust/detail/host/adjacent_difference.h>
#include <thrust/detail/device/adjacent_difference.h>

namespace thrust
{
namespace detail
{
namespace dispatch
{

///////////////
// Host Path //
///////////////
template <class InputIterator, class OutputIterator, class BinaryFunction>
OutputIterator adjacent_difference(InputIterator first, InputIterator last,
                                   OutputIterator result,
                                   BinaryFunction binary_op,
                                   thrust::host_space_tag,
                                   thrust::host_space_tag)
{
    return thrust::detail::host::adjacent_difference(first, last, result, binary_op);
}


/////////////////
// Device Path //
/////////////////
template <class InputIterator, class OutputIterator, class BinaryFunction>
OutputIterator adjacent_difference(InputIterator first, InputIterator last,
                                   OutputIterator result,
                                   BinaryFunction binary_op,
                                   thrust::device_space_tag,
                                   thrust::device_space_tag)
{
    return thrust::detail::device::adjacent_difference(first, last, result, binary_op);
}

} // end namespace dispatch
} // end namespace detail
} // end namespace thrust

