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

template<typename InputIterator,
         typename UnaryFunction,
         typename Space>
  void fill(InputIterator first,
            InputIterator last,
            UnaryFunction f,
            Space)
{
    // generic backend
    thrust::detail::device::generic::fill(first, last, f);
}

template<typename InputIterator,
         typename UnaryFunction>
  void fill(InputIterator first,
            InputIterator last,
            UnaryFunction f,
            thrust::detail::cuda_device_space_tag)
{
    // refinement for the CUDA backend 
    thrust::detail::device::cuda::fill(first, last, f);
}

} // end dispatch

} // end device

} // end detail

} // end thrust

