/*
 *  Copyright 2008-2012 NVIDIA Corporation
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

struct device_properties_t
{
  // mirror the type and spelling of cudaDeviceProp's members
  // keep these alphabetized
  int    major;
  int    maxGridSize[3];
  int    maxThreadsPerBlock;
  int    maxThreadsPerMultiProcessor;
  int    minor;
  int    multiProcessorCount;
  int    regsPerBlock;
  size_t sharedMemPerBlock;
  int    warpSize;
};

struct function_attributes_t
{
  // mirror the type and spelling of cudaFuncAttributes' members
  // keep these alphabetized
  size_t constSizeBytes;
  size_t localSizeBytes;
  int    maxThreadsPerBlock;
  int    numRegs;
  size_t sharedSizeBytes;
};

/*! Returns a copy of the device_properties_t structure
 *  that is associated with a given device.
 */
inline device_properties_t device_properties(int device_id);

/*! Returns a copy of the device_properties_t structure
 *  that is associated with the current device.
 */
inline device_properties_t device_properties(void);

/*! Returns a copy of the function_attributes_t structure
 *  that is associated with a given __global__ function
 */
template <typename KernelFunction>
inline function_attributes_t function_attributes(KernelFunction kernel);

/*! Returns the compute capability of a device in integer format.
 *  For example, returns 10 for sm_10 and 21 for sm_21
 *  \return The compute capability as an integer
 */
inline size_t compute_capability(const device_properties_t &properties);
inline size_t compute_capability(void);


/*! Returns the maximum number of blocks (of a particular kernel)
 *  that can be resident on a single multiprocessor.
 */
inline size_t max_active_blocks_per_multiprocessor(const device_properties_t&   properties,
                                                   const function_attributes_t& attributes,
                                                   const size_t CTA_SIZE,
                                                   const size_t dynamic_smem_bytes);

/*! Returns a pair (block_size,blocks_per_multiprocessor)
 *  where block_size is a valid block size chosen by
 *  a heuristic and blocks_per_multiprocessor is the 
 *  maximum number of such blocks that can execute on
 *  a (streaming) multiprocessor at once.
 *
 *  \param properties CUDA device properties
 *  \param attributes CUDA function attributes
 */
inline thrust::pair<size_t,size_t> default_block_configuration(const device_properties_t&   properties,
                                                               const function_attributes_t& attributes);

/*! Returns a pair (block_size,blocks_per_multiprocessor)
 *  where block_size is a valid block size chosen by
 *  a heuristic and blocks_per_multiprocessor is the 
 *  maximum number of such blocks that can execute on
 *  a (streaming) multiprocessor at once.
 *
 *  \param properties CUDA device properties
 *  \param attributes CUDA function attributes
 *  \param UnaryFunction Mapping from block size to (dynamic) shared memory allocation
 */
template <typename UnaryFunction>
thrust::pair<size_t,size_t> default_block_configuration(const device_properties_t   &properties,
                                                        const function_attributes_t &attributes,
                                                        UnaryFunction block_size_to_smem_size);


/*! Returns the maximum amount of dynamic shared memory each block
 *  can utilize without reducing thread occupancy.
 *
 *  \param properties CUDA device properties
 *  \param attributes CUDA function attributes
 *  \param blocks_per_processor Number of blocks per streaming multiprocessor
 */
inline size_t proportional_smem_allocation(const device_properties_t   &properties,
                                           const function_attributes_t &attributes,
                                           size_t blocks_per_processor);


// TODO try to eliminate following functions

template <typename KernelFunction>
size_t max_active_blocks(KernelFunction kernel, const size_t CTA_SIZE, const size_t dynamic_smem_bytes);


/*! This function returns the block size that achieves the highest
 *  occupancy for a particular kernel & device.
 */
inline size_t max_blocksize_with_highest_occupancy(const device_properties_t   &properties,
                                                   const function_attributes_t &attributes,
                                                   size_t dynamic_smem_bytes_per_thread = 0);

template <typename KernelFunction>
size_t max_blocksize_with_highest_occupancy(KernelFunction kernel, size_t dynamic_smem_bytes_per_thread = 0);

/*! This function returns the maximum block size for a given kernel and device.
 */
inline size_t max_blocksize(const device_properties_t   &properties,
                            const function_attributes_t &attributes,
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

