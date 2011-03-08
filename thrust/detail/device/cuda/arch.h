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


/*! \file arch.h
 *  \brief Defines the interface to functions
 *         providing introspection into the architecture
 *         of CUDA devices.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/tuple.h>

// #include this for size_t
#include <cstddef>

// avoid #including a header,
// just provide forward declarations
struct cudaDeviceProp;
struct cudaFuncAttributes;

namespace thrust
{

namespace detail
{

namespace device
{

namespace cuda
{

namespace arch
{

  
/*! This function returns the compute capability of a device.
 *  For example, returns 10 for sm_10 and 21 for sm_21
 *  \return The compute capability as an integer
 */

inline size_t compute_capability(const cudaDeviceProp &properties);
inline size_t compute_capability(void);

/*! This function returns the number of streaming
 *  multiprocessors available for processing.
 *  \return The number of SMs available.
 */
inline size_t num_multiprocessors(const cudaDeviceProp&);
inline size_t num_multiprocessors(void);

/*! This function returns the maximum number of
 *  threads active on a single multiprocessor.
 *  \return The maximum number of threads active on
 *          a single multiprocessor.
 */
inline size_t max_active_threads_per_multiprocessor(const cudaDeviceProp&);
inline size_t max_active_threads_per_multiprocessor(void);

/*! This function returns the maximum number of
 *  active threads allowed across all multiprocessors.
 *  \return The maximum number of active threads.
 */
inline size_t max_active_threads(const cudaDeviceProp&);
inline size_t max_active_threads(void);

/*! This function returns the maximum size of each
 *  dimension of a grid of thread blocks.
 *  \return A 3-tuple containing, for each dimension, the maximum
 *          size of a grid of thread blocks.
 */
inline thrust::tuple<unsigned int,unsigned int,unsigned int> max_grid_dimensions(const cudaDeviceProp&);
inline thrust::tuple<unsigned int,unsigned int,unsigned int> max_grid_dimensions(void);

/*! This function returns the maximum number of
 *  blocks (of a particular kernel) that can be resident on
 *  a single multiprocessor.
 */
inline size_t max_active_blocks_per_multiprocessor(const cudaDeviceProp& properties,
                                                   const cudaFuncAttributes& attributes,
                                                   const size_t CTA_SIZE,
                                                   const size_t dynamic_smem_bytes);

template <typename KernelFunction>
size_t max_active_blocks(KernelFunction kernel, const size_t CTA_SIZE, const size_t dynamic_smem_bytes);


/*! This function returns the block size that achieves the highest
 *  occupancy for a particular kernel & device.
 */
inline size_t max_blocksize_with_highest_occupancy(const cudaDeviceProp& properties,
                                                   const cudaFuncAttributes& attributes,
                                                   size_t dynamic_smem_bytes_per_thread = 0);

template <typename KernelFunction>
size_t max_blocksize_with_highest_occupancy(KernelFunction kernel, size_t dynamic_smem_bytes_per_thread = 0);

/*! This function returns the maximum block size for a given kernel and device.
 */
inline size_t max_blocksize(const cudaDeviceProp& properties,
                            const cudaFuncAttributes& attributes,
                            size_t dynamic_smem_bytes_per_thread = 0);

template <typename KernelFunction>
size_t max_blocksize(KernelFunction kernel, size_t dynamic_smem_bytes_per_thread = 0);

template<typename KernelFunction, typename UnaryFunction>
size_t max_blocksize_subject_to_smem_usage(KernelFunction kernel, UnaryFunction blocksize_to_dynamic_smem_usage);

} // end namespace arch
} // end namespace cuda
} // end namespace device
} // end namespace detail
} // end namespace thrust

#include <thrust/detail/device/cuda/arch.inl>

