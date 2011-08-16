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


/*! \file arch.inl
 *  \brief Inline file for arch.h.
 */

#include <thrust/detail/config.h>

// guard this file, which depends on CUDA, from compilers which aren't nvcc
#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC

#include <cuda_runtime_api.h>

#include <map>
#include <string>     // TODO remove this?
#include <algorithm>

#include <thrust/system_error.h>
#include <thrust/system/cuda_error.h>
#include <thrust/detail/util/blocking.h>
#include <thrust/detail/backend/cuda/detail/launch_closure.h>

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
namespace detail
{

// granularity of shared memory allocation
inline size_t smem_allocation_unit(const cudaDeviceProp&)
{
  return 512;
}

// granularity of register allocation
inline size_t reg_allocation_unit(const cudaDeviceProp& properties)
{
  return (properties.major < 2 && properties.minor < 2) ? 256 : 512;
}

// granularity of warp allocation
inline size_t warp_allocation_multiple(const cudaDeviceProp&)
{
  return 2;
}

inline size_t max_blocks_per_multiprocessor(const cudaDeviceProp&)
{
  return 8;
}

template <typename T>
struct zero_function
{
  T operator()(T)
  {
    return 0;
  }
};

} // end namespace detail


inline const cudaDeviceProp& device_properties(int device_id)
{
  // cache the result of the introspection call because it is expensive
  static std::map<int,cudaDeviceProp> properties_map;

  // search the cache for the properties
  std::map<int,cudaDeviceProp>::const_iterator iter = properties_map.find(device_id);

  if(iter == properties_map.end())
  {
    cudaDeviceProp properties;

    // the properties weren't found, ask the runtime to generate them
    cudaError_t error = cudaGetDeviceProperties(&properties, device_id);

    if(error)
      throw thrust::system_error(error, thrust::cuda_category());

    // insert the new entry and return a reference
    return properties_map[device_id] = properties;
  }
  else
  {
    // return the cached value
    return iter->second;
  }
}

inline const cudaDeviceProp& device_properties(void)
{
  int device_id = -1;

  cudaError_t error = cudaGetDevice(&device_id);

  if(error)
    throw thrust::system_error(error, thrust::cuda_category());

  if(device_id < 0)
    throw thrust::system_error(cudaErrorNoDevice, thrust::cuda_category());

  return device_properties(device_id);
}

template <typename KernelFunction>
inline const cudaFuncAttributes& function_attributes(KernelFunction kernel)
{
  typedef void (*fun_ptr_type)();

  // cache the result of the introspection call because it is expensive
  // cache fun_ptr_type rather than KernelFunction to avoid problems with long names on MSVC 2005
  static std::map<fun_ptr_type,cudaFuncAttributes> attributes_map;

  fun_ptr_type fun_ptr = reinterpret_cast<fun_ptr_type>(kernel);

  // search the cache for the attributes
  typename std::map<fun_ptr_type,cudaFuncAttributes>::const_iterator iter = attributes_map.find(fun_ptr);

  if(iter == attributes_map.end())
  {
    cudaFuncAttributes attributes;

    // the attributes weren't found, ask the runtime to generate them
    cudaError_t error = cudaFuncGetAttributes(&attributes, kernel);
  
    if(error)
    {
      throw thrust::system_error(error, thrust::cuda_category());
    }

    // insert the new entry and return a reference
    return attributes_map[fun_ptr] = attributes;
  }
  else
  {
    // return the cached value
    return iter->second;
  }
}

inline size_t compute_capability(const cudaDeviceProp &properties)
{
  return 10 * properties.major + properties.minor;
}

inline size_t compute_capability(void)
{
  return compute_capability(device_properties());
}


inline size_t max_active_blocks_per_multiprocessor(const cudaDeviceProp&     properties,
                                                   const cudaFuncAttributes& attributes,
                                                   size_t CTA_SIZE,
                                                   size_t dynamic_smem_bytes)
{
  // Determine the maximum number of CTAs that can be run simultaneously per SM
  // This is equivalent to the calculation done in the CUDA Occupancy Calculator spreadsheet
  const size_t regAllocationUnit      = detail::reg_allocation_unit(properties);
  const size_t warpAllocationMultiple = detail::warp_allocation_multiple(properties);
  const size_t smemAllocationUnit     = detail::smem_allocation_unit(properties);
  const size_t maxThreadsPerSM        = properties.maxThreadsPerMultiProcessor;  // 768, 1024, 1536, etc.
  const size_t maxBlocksPerSM         = detail::max_blocks_per_multiprocessor(properties);

  // Number of warps (round up to nearest whole multiple of warp size & warp allocation multiple)
  const size_t numWarps = thrust::detail::util::round_i(thrust::detail::util::divide_ri(CTA_SIZE, properties.warpSize), warpAllocationMultiple);

  // Number of regs is regs per thread times number of warps times warp size
  const size_t regsPerCTA = thrust::detail::util::round_i(attributes.numRegs * properties.warpSize * numWarps, regAllocationUnit);

  const size_t smemBytes  = attributes.sharedSizeBytes + dynamic_smem_bytes;
  const size_t smemPerCTA = thrust::detail::util::round_i(smemBytes, smemAllocationUnit);

  const size_t ctaLimitRegs    = regsPerCTA > 0 ? properties.regsPerBlock      / regsPerCTA : maxBlocksPerSM;
  const size_t ctaLimitSMem    = smemPerCTA > 0 ? properties.sharedMemPerBlock / smemPerCTA : maxBlocksPerSM;
  const size_t ctaLimitThreads =                  maxThreadsPerSM              / CTA_SIZE;

  return std::min<size_t>(ctaLimitRegs, std::min<size_t>(ctaLimitSMem, std::min<size_t>(ctaLimitThreads, maxBlocksPerSM)));
}



inline thrust::pair<size_t,size_t> default_block_configuration(const cudaDeviceProp&     properties,
                                                               const cudaFuncAttributes& attributes)
{
  return default_block_configuration(properties, attributes, detail::zero_function<size_t>());
}

template <typename UnaryFunction>
thrust::pair<size_t,size_t> default_block_configuration(const cudaDeviceProp&     properties,
                                                        const cudaFuncAttributes& attributes,
                                                        UnaryFunction block_size_to_smem_size)
{
  size_t max_occupancy      = properties.maxThreadsPerMultiProcessor;
  size_t largest_blocksize  = (std::min)(properties.maxThreadsPerBlock, attributes.maxThreadsPerBlock);
  size_t granularity        = properties.warpSize;
  size_t max_blocksize      = 0;
  size_t highest_occupancy  = 0;

  for(size_t blocksize = largest_blocksize; blocksize != 0; blocksize -= granularity)
  {
    size_t occupancy = blocksize * max_active_blocks_per_multiprocessor(properties, attributes, blocksize, block_size_to_smem_size(blocksize));

    if (occupancy > highest_occupancy)
    {
      max_blocksize = blocksize;
      highest_occupancy = occupancy;
    }

    // early out, can't do better
    if (highest_occupancy == max_occupancy)
      break;
  }

  return thrust::make_pair(max_blocksize, max_occupancy / max_blocksize);
}


inline size_t proportional_smem_allocation(const cudaDeviceProp& properties,
                                           const cudaFuncAttributes& attributes,
                                           size_t blocks_per_processor)
{
  size_t smem_per_processor    = properties.sharedMemPerBlock;
  size_t smem_allocation_unit  = thrust::detail::backend::cuda::arch::detail::smem_allocation_unit(properties);

  size_t total_smem_per_block  = thrust::detail::util::round_z(smem_per_processor / blocks_per_processor, smem_allocation_unit);
  size_t static_smem_per_block = attributes.sharedSizeBytes;
  
  return total_smem_per_block - static_smem_per_block;
}


// TODO try to eliminate following functions

template <typename KernelFunction>
size_t max_active_blocks(KernelFunction kernel, const size_t CTA_SIZE, const size_t dynamic_smem_bytes)
{
  const cudaDeviceProp&     properties = device_properties();
  const cudaFuncAttributes& attributes = function_attributes(kernel);

  return properties.multiProcessorCount * max_active_blocks_per_multiprocessor(properties, attributes, CTA_SIZE, dynamic_smem_bytes);
}

size_t max_blocksize_with_highest_occupancy(const cudaDeviceProp&     properties,
                                            const cudaFuncAttributes& attributes,
                                            size_t dynamic_smem_bytes_per_thread)
{
  size_t max_occupancy      = properties.maxThreadsPerMultiProcessor;
  size_t largest_blocksize  = (std::min)(properties.maxThreadsPerBlock, attributes.maxThreadsPerBlock);
  size_t granularity        = properties.warpSize;
  size_t max_blocksize      = 0;
  size_t highest_occupancy  = 0;

  for(size_t blocksize = largest_blocksize; blocksize != 0; blocksize -= granularity)
  {
    size_t occupancy = blocksize * max_active_blocks_per_multiprocessor(properties, attributes, blocksize, dynamic_smem_bytes_per_thread * blocksize);

    if (occupancy > highest_occupancy)
    {
      max_blocksize = blocksize;
      highest_occupancy = occupancy;
    }

    // early out, can't do better
    if (highest_occupancy == max_occupancy)
      return max_blocksize;
  }

  return max_blocksize;
}

template <typename KernelFunction>
size_t max_blocksize_with_highest_occupancy(KernelFunction kernel, size_t dynamic_smem_bytes_per_thread)
{
  const cudaDeviceProp&     properties = device_properties();
  const cudaFuncAttributes& attributes = function_attributes(kernel);
  
  return max_blocksize_with_highest_occupancy(properties, attributes, dynamic_smem_bytes_per_thread);
}


// TODO unify this with max_blocksize_with_highest_occupancy
size_t max_blocksize(const cudaDeviceProp&     properties,
                     const cudaFuncAttributes& attributes,
                     size_t dynamic_smem_bytes_per_thread)
{
  size_t largest_blocksize  = (std::min)(properties.maxThreadsPerBlock, attributes.maxThreadsPerBlock);

  size_t granularity = properties.warpSize;

  for(size_t blocksize = largest_blocksize; blocksize != 0; blocksize -= granularity)
  {
    if(0 < max_active_blocks_per_multiprocessor(properties, attributes, blocksize, dynamic_smem_bytes_per_thread * blocksize))
      return blocksize;
  }

  return 0;
}

template <typename KernelFunction>
size_t max_blocksize(KernelFunction kernel, size_t dynamic_smem_bytes_per_thread)
{
  const cudaDeviceProp&     properties = device_properties();
  const cudaFuncAttributes& attributes = function_attributes(kernel);

  return max_blocksize(properties, attributes, dynamic_smem_bytes_per_thread);
}

template<typename UnaryFunction>
size_t max_blocksize_subject_to_smem_usage(const cudaDeviceProp&     properties,
                                           const cudaFuncAttributes& attributes,
                                           UnaryFunction blocksize_to_dynamic_smem_usage)
{
  size_t largest_blocksize = (std::min)(properties.maxThreadsPerBlock, attributes.maxThreadsPerBlock);
  size_t granularity = properties.warpSize;
  
  for(int blocksize = largest_blocksize; blocksize > 0; blocksize -= granularity)
  {
    size_t total_smem_usage = blocksize_to_dynamic_smem_usage(blocksize) + attributes.sharedSizeBytes;

    if(total_smem_usage <= properties.sharedMemPerBlock)
    {
      return blocksize;
    }
  }

  return 0;
}

template<typename KernelFunction, typename UnaryFunction>
size_t max_blocksize_subject_to_smem_usage(KernelFunction kernel, UnaryFunction blocksize_to_dynamic_smem_usage)
{
  const cudaDeviceProp&     properties = device_properties();
  const cudaFuncAttributes& attributes = function_attributes(kernel);
  
  return max_blocksize_subject_to_smem_usage(properties, attributes, blocksize_to_dynamic_smem_usage);
}

} // end namespace arch
} // end namespace cuda
} // end namespace backend
} // end namespace detail
} // end namespace thrust

#endif // THRUST_DEVICE_COMPILER_NVCC

