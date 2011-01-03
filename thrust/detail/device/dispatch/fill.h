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

#include <thrust/iterator/iterator_traits.h>
#include <thrust/detail/device/cuda/fill.h>
#include <thrust/detail/device/generic/fill.h>

namespace thrust
{

namespace detail
{

namespace device
{

namespace dispatch
{

template<typename OutputIterator,
         typename Size,
         typename T,
         typename Space>
  OutputIterator fill_n(OutputIterator first,
                        Size n,
                        const T &value,
                        Space)
{
  // generic backend
  return thrust::detail::device::generic::fill_n(first, n, value);
}

template<typename OutputIterator,
         typename Size,
         typename T>
  OutputIterator fill_n(OutputIterator first,
                        Size n,
                        const T &value,
                        thrust::detail::cuda_device_space_tag)
{
  // refinement for the CUDA backend 
  return thrust::detail::device::cuda::fill_n(first, n, value);
}

} // end dispatch

} // end device

} // end detail

} // end thrust

