/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in ccudaliance with the License.
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

/*! \file thrust/system/cuda/vector.h
 *  \brief A dynamically-sizable array of elements which reside in memory available to
 *         Thrust's CUDA system.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/system/cuda/memory.h>
#include <thrust/detail/vector_base.h>
#include <vector>

namespace thrust
{

// forward declaration of host_vector
template<typename T, typename Allocator> class host_vector;

namespace cuda_cub
{

/*! \p cuda_bulk::vector is a container that supports random access to elements,
 *  constant time removal of elements at the end, and linear time insertion
 *  and removal of elements at the beginning or in the middle. The number of
 *  elements in a \p cuda_bulk::vector may vary dynamically; memory management is
 *  automatic. The elements contained in a \p cuda_bulk::vector reside in memory
 *  available to the \p cuda_bulk system.
 *
 *  \tparam T The element type of the \p cuda_bulk::vector.
 *  \tparam Allocator The allocator type of the \p cuda_bulk::vector. Defaults to \p cuda_bulk::allocator.
 *
 *  \see http://www.sgi.com/tech/stl/Vector.html
 *  \see host_vector For the documentation of the complete interface which is
 *                   shared by \p cuda_bulk::vector
 *  \see device_vector
 */
template<typename T, typename Allocator = allocator<T> >
using vector = thrust::detail::vector_base<T, Allocator>;

} // end cuda_cub

// alias system::cuda_bulk names at top-level
namespace cuda
{

using thrust::cuda_cub::vector;

} // end cuda_bulk

namespace system {
namespace cuda {
using thrust::cuda_cub::vector;
}
}

} // end thrust
