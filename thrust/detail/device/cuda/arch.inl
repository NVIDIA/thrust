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

#include <string>
#include <algorithm>

#include <thrust/detail/util/blocking.h>
#include <thrust/system_error.h>
#include <thrust/system/cuda_error.h>
#include <map>

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
namespace detail
{

inline void checked_get_current_device_properties(cudaDeviceProp &properties)
{
  int current_device = -1;

  cudaError_t error = cudaGetDevice(&current_device);

  if(error)
  {
    throw thrust::system_error(error, thrust::cuda_category());
  }

  if(current_device < 0)
    throw thrust::system_error(cudaErrorNoDevice, thrust::cuda_category());

  // cache the result of the introspection call because it is expensive
  static std::map<int,cudaDeviceProp> properties_map;

  // search the cache for the properties
  std::map<int,cudaDeviceProp>::const_iterator iter = properties_map.find(current_device);

  if(iter == properties_map.end())
  {
    // the properties weren't found, ask the runtime to generate them
    error = cudaGetDeviceProperties(&properties, current_device);

    if(error)
    {
      throw thrust::system_error(error, thrust::cuda_category());
    }

    // insert the new entry
    properties_map[current_device] = properties;
  } // end if
  else
  {
    // use the cached value
    properties = iter->second;
  } // end else
} // end checked_get_current_device_properties()

template <typename KernelFunction>
void checked_get_function_attributes(cudaFuncAttributes& attributes, KernelFunction kernel)
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
    // the attributes weren't found, ask the runtime to generate them
    cudaError_t error = cudaFuncGetAttributes(&attributes, kernel);
  
    if(error)
    {
      throw thrust::system_error(error, thrust::cuda_category());
    }

    // insert the new entry
    attributes_map[fun_ptr] = attributes;
  } // end if
  else
  {
    // use the cached value
    attributes = iter->second;
  } // end else
} // end checked_get_function_attributes()

} // end detail

size_t compute_capability(const cudaDeviceProp &properties)
{
  return 10 * properties.major + properties.minor;
} // end compute_capability()

size_t num_multiprocessors(const cudaDeviceProp& properties)
{
    return properties.multiProcessorCount;
} // end num_multiprocessors()


size_t max_active_threads_per_multiprocessor(const cudaDeviceProp& properties)
{
    // index this array by [major, minor] revision
    // \see NVIDIA_CUDA_Programming_Guide_3.0.pdf p 140
    static const size_t max_active_threads_by_compute_capability[3][4] = \
        {{     0,    0,    0,    0},
         {   768,  768, 1024, 1024},
         {  1536, 1536, 1536, 1536}};

    // produce valid results for new, unknown devices
    if (properties.major > 2 || properties.minor > 3)
        return max_active_threads_by_compute_capability[2][3];
    else
        return max_active_threads_by_compute_capability[properties.major][properties.minor];
} // end max_active_threads_per_multiprocessor()


size_t max_active_threads(const cudaDeviceProp& properties)
{
    return num_multiprocessors(properties) * max_active_threads_per_multiprocessor(properties);
} // end max_active_threads()


thrust::tuple<unsigned int,unsigned int,unsigned int> max_grid_dimensions(const cudaDeviceProp& properties)
{
    return thrust::make_tuple(properties.maxGridSize[0], properties.maxGridSize[1], properties.maxGridSize[2]);
} // end max_grid_dimensions()
  

size_t max_active_blocks_per_multiprocessor(const cudaDeviceProp& properties,
                                            const cudaFuncAttributes& attributes,
                                            size_t CTA_SIZE,
                                            size_t dynamic_smem_bytes)
{
    // Determine the maximum number of CTAs that can be run simultaneously per SM
    // This is equivalent to the calculation done in the CUDA Occupancy Calculator spreadsheet
    const size_t regAllocationUnit = (properties.major < 2 && properties.minor < 2) ? 256 : 512; // in registers
    const size_t warpAllocationMultiple = 2;
    const size_t smemAllocationUnit = 512;                                                 // in bytes
    const size_t maxThreadsPerSM = max_active_threads_per_multiprocessor(properties);      // 768, 1024, etc.
    const size_t maxBlocksPerSM = 8;

    // Number of warps (round up to nearest whole multiple of warp size & warp allocation multiple)
    const size_t numWarps = thrust::detail::util::round_i(thrust::detail::util::divide_ri(CTA_SIZE, properties.warpSize), warpAllocationMultiple);

    // Number of regs is regs per thread times number of warps times warp size
    const size_t regsPerCTA = thrust::detail::util::round_i(attributes.numRegs * properties.warpSize * numWarps, regAllocationUnit);

    const size_t smemBytes  = attributes.sharedSizeBytes + dynamic_smem_bytes;
    const size_t smemPerCTA = thrust::detail::util::round_i(smemBytes, smemAllocationUnit);

    const size_t ctaLimitRegs    = regsPerCTA > 0 ? properties.regsPerBlock / regsPerCTA : maxBlocksPerSM;
    const size_t ctaLimitSMem    = smemPerCTA > 0 ? properties.sharedMemPerBlock   / smemPerCTA : maxBlocksPerSM;
    const size_t ctaLimitThreads =                  maxThreadsPerSM                / CTA_SIZE;

    return std::min<size_t>(ctaLimitRegs, std::min<size_t>(ctaLimitSMem, std::min<size_t>(ctaLimitThreads, maxBlocksPerSM)));
}



// Functions that query the runtime for device properties
size_t compute_capability(void)
{
    cudaDeviceProp properties;  
    detail::checked_get_current_device_properties(properties);
    return compute_capability(properties);
} // end compute_capability()


size_t num_multiprocessors(void)
{
    cudaDeviceProp properties;  
    detail::checked_get_current_device_properties(properties);
    return num_multiprocessors(properties);
} // end num_multiprocessors()

size_t max_active_threads_per_multiprocessor(void)
{
    cudaDeviceProp properties;  
    detail::checked_get_current_device_properties(properties);
    return max_active_threads_per_multiprocessor(properties);
}

size_t max_active_threads(void)
{
    cudaDeviceProp properties;  
    detail::checked_get_current_device_properties(properties);
    return max_active_threads(properties);
}

thrust::tuple<unsigned int,unsigned int,unsigned int> max_grid_dimensions(void)
{
    cudaDeviceProp properties;  
    detail::checked_get_current_device_properties(properties);
    return max_grid_dimensions(properties);
}

template <typename KernelFunction>
size_t max_active_blocks(KernelFunction kernel, const size_t CTA_SIZE, const size_t dynamic_smem_bytes)
{
    cudaDeviceProp properties;  
    detail::checked_get_current_device_properties(properties);

    cudaFuncAttributes attributes;
    detail::checked_get_function_attributes(attributes, kernel);

    return num_multiprocessors(properties) * max_active_blocks_per_multiprocessor(properties, attributes, CTA_SIZE, dynamic_smem_bytes);
}

size_t max_blocksize_with_highest_occupancy(const cudaDeviceProp& properties,
                                            const cudaFuncAttributes& attributes,
                                            size_t dynamic_smem_bytes_per_thread)
{
    size_t max_occupancy = max_active_threads_per_multiprocessor(properties);

    size_t largest_blocksize  = (std::min)(properties.maxThreadsPerBlock, attributes.maxThreadsPerBlock);

    size_t granularity        = properties.warpSize;

    size_t max_blocksize     = 0;
    size_t highest_occupancy = 0;

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
    cudaDeviceProp properties;  
    detail::checked_get_current_device_properties(properties);

    cudaFuncAttributes attributes;
    detail::checked_get_function_attributes(attributes, kernel);

    return max_blocksize_with_highest_occupancy(properties, attributes, dynamic_smem_bytes_per_thread);
}


// TODO unify this with max_blocksize_with_highest_occupancy
size_t max_blocksize(const cudaDeviceProp& properties,
                     const cudaFuncAttributes& attributes,
                     size_t dynamic_smem_bytes_per_thread)
{
    size_t largest_blocksize  = (std::min)(properties.maxThreadsPerBlock, attributes.maxThreadsPerBlock);

    // TODO eliminate this constant (i assume this is warp_size)
    size_t granularity        = 32;

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
    cudaDeviceProp properties;  
    detail::checked_get_current_device_properties(properties);

    cudaFuncAttributes attributes;
    detail::checked_get_function_attributes(attributes, kernel);

    return max_blocksize(properties, attributes, dynamic_smem_bytes_per_thread);
}

template<typename UnaryFunction>
size_t max_blocksize_subject_to_smem_usage(const cudaDeviceProp& properties,
                                           const cudaFuncAttributes& attributes,
                                           UnaryFunction blocksize_to_dynamic_smem_usage)
{
    size_t largest_blocksize = (std::min)(properties.maxThreadsPerBlock, attributes.maxThreadsPerBlock);
    
    // XXX eliminate this constant (i assume this is warp_size)
    size_t granularity = 32;

    for(int blocksize = largest_blocksize;
        blocksize > 0;
        blocksize -= granularity)
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
    cudaDeviceProp properties;  
    detail::checked_get_current_device_properties(properties);

    cudaFuncAttributes attributes;
    detail::checked_get_function_attributes(attributes, kernel);

    return max_blocksize_subject_to_smem_usage(properties, attributes, blocksize_to_dynamic_smem_usage);
}

} // end namespace arch
} // end namespace cuda
} // end namespace device
} // end namespace detail
} // end namespace thrust

#endif // THRUST_DEVICE_COMPILER_NVCC

