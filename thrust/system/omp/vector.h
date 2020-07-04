/*
 *  Copyright 2008-2013 NVIDIA Corporation
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

/*! \file thrust/system/omp/vector.h
 *  \brief A dynamically-sizable array of elements which reside in memory available to
 *         Thrust's OpenMP system.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/system/omp/memory.h>
#include <thrust/detail/vector_base.h>
#include <vector>

namespace thrust
{

// forward declaration of host_vector
// XXX why is this here? it doesn't seem necessary for anything below
template<typename T, typename Allocator> class host_vector;

namespace system
{
namespace omp
{

/*! \p omp::vector is a container that supports random access to elements,
 *  constant time removal of elements at the end, and linear time insertion
 *  and removal of elements at the beginning or in the middle. The number of
 *  elements in a \p omp::vector may vary dynamically; memory management is
 *  automatic. The elements contained in an \p omp::vector reside in memory
 *  available to the \p omp system.
 *
 *  \tparam T The element type of the \p omp::vector.
 *  \tparam Allocator The allocator type of the \p omp::vector. Defaults to \p omp::allocator.
 *
 *  \see http://www.sgi.com/tech/stl/Vector.html
 *  \see host_vector For the documentation of the complete interface which is
 *                   shared by \p omp::vector
 *  \see device_vector
 */
template<typename T, typename Allocator = allocator<T> >
using vector = thrust::detail::vector_base<T, Allocator>;

} // end omp
} // end system

// alias system::omp names at top-level
namespace omp
{

using thrust::system::omp::vector;

} // end omp

} // end thrust
