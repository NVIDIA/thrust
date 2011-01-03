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


/*! \file uninitialized_copy.h
 *  \brief Defines the interface to the dispatch
 *         layer of the uninitialized_copy function.
 */

#pragma once

#include <memory>
#include <thrust/detail/device/uninitialized_copy.h>
#include <thrust/iterator/iterator_traits.h>


namespace thrust
{

namespace detail
{

namespace dispatch
{

template<typename InputIterator,
         typename ForwardIterator>
  ForwardIterator uninitialized_copy(InputIterator first,
                                     InputIterator last,
                                     ForwardIterator result,
                                     thrust::host_space_tag)
{
  return std::uninitialized_copy(first, last, result);
} // end uninitialized_copy()


template<typename InputIterator,
         typename ForwardIterator>
  ForwardIterator uninitialized_copy(InputIterator first,
                                     InputIterator last,
                                     ForwardIterator result,
                                     thrust::device_space_tag)
{
  return thrust::detail::device::uninitialized_copy(first, last, result);
} // end uninitialized_copy()


} // end dispatch

} // end detail

} // end thrust

