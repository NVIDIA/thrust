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
namespace backend
{
namespace cuda
{
namespace arch
{

/*! Returns a reference to the cudaDeviceProp structure
 *  that is associated with a given device.
 */
inline const cudaDeviceProp& device_properties(int device_id);

/*! Returns a reference to the cudaDeviceProp structure
 *  that is associated with the current device.
 */
inline const cudaDeviceProp& device_properties(void);

/*! Returns a reference to the cudaFuncAttributes structure
 *  that is associated with a given __global__ function
 */
template <typename KernelFunction>
inline const cudaFuncAttributes& function_attributes(KernelFunction kernel);
  
/*! Returns the compute capability of a device in integer format.
 *  For example, returns 10 for sm_10 and 21 for sm_21
 *  \return The compute capability as an integer
 */
inline size_t compute_capability(const cudaDeviceProp &properties);
inline size_t compute_capability(void);


/*! Returns the maximum number of blocks (of a particular kernel)
 *  that can be resident on a single multiprocessor.
 */
inline size_t max_active_blocks_per_multiprocessor(const cudaDeviceProp&     properties,
                                                   const cudaFuncAttributes& attributes,
                                                   const size_t CTA_SIZE,
                                                   const size_t dynamic_smem_bytes);


// TODO try to eliminate following functions

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
inline size_t max_blocksize(const cudaDeviceProp&     properties,
                            const cudaFuncAttributes& attributes,
                            size_t dynamic_smem_bytes_per_thread = 0);

template <typename KernelFunction>
size_t max_blocksize(KernelFunction kernel, size_t dynamic_smem_bytes_per_thread = 0);

template<typename KernelFunction, typename UnaryFunction>
size_t max_blocksize_subject_to_smem_usage(KernelFunction kernel, UnaryFunction blocksize_to_dynamic_smem_usage);

} // end namespace arch
} // end namespace cuda
} // end namespace backend
} // end namespace detail
} // end namespace thrust

#include <thrust/detail/backend/cuda/arch.inl>

