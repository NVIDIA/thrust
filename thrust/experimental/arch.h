/*
 *  Copyright 2008-2009 NVIDIA Corporation
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


/*! \file arch.h
 *  \brief Defines the interface to functions
 *         providing introspection into the architecture
 *         of CUDA devices.
 */

#pragma once

#include <thrust/detail/config.h>

// #include this for size_t
#include <cstddef>

// #include this for dim3
#include <vector_types.h>

namespace thrust
{

namespace experimental
{

namespace arch
{

/*! This function returns the number of streaming
 *  multiprocessors available for processing.
 *  \return The number of SMs available.
 */
inline size_t num_multiprocessors(void);

/*! This function returns the maximum number of
 *  threads active on a single multiprocessor.
 *  \return The maximum number of threads active on
 *          a single multiprocessor.
 */
inline size_t max_active_threads_per_multiprocessor(void);

/*! This function returns the maximum number of
 *  active threads allowed across all multiprocessors.
 *  \return The maximum number of active threads.
 */
inline size_t max_active_threads(void);

/*! This function returns the maximum size of each
 *  dimension of a grid of thread blocks.
 *  \return A dim3 containing, for each dimension, the maximum
 *          size of a grid of thread blocks.
 */
inline dim3 max_grid_dimensions(void);

}; // end arch

}; // end experimental

}; // end thrust

#include <thrust/experimental/arch.inl>

