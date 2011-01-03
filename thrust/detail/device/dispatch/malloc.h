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

#pragma once

#include <thrust/device_ptr.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/detail/device/generic/malloc.h>
#include <thrust/detail/device/cuda/malloc.h>

namespace thrust
{

namespace detail
{

namespace device
{

// XXX forward declaration to WAR circular #inclusion
namespace cuda
{

template<unsigned int>
thrust::device_ptr<void> malloc(const std::size_t n);

} // end cuda

namespace dispatch
{

template<unsigned int DummyParameterToAvoidInstantiation>
thrust::device_ptr<void> malloc(const std::size_t n,
                                thrust::device_space_tag)
{
  return thrust::detail::device::generic::malloc<0>(n);
} // end malloc()

template<unsigned int DummyParameterToAvoidInstantiation>
thrust::device_ptr<void> malloc(const std::size_t n,
                                thrust::detail::cuda_device_space_tag)
{
  return thrust::detail::device::cuda::malloc<0>(n);
} // end malloc()

} // end dispatch

} // end device

} // end detail

} // end thrust

