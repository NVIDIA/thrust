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


/*! \file arch.inl
 *  \brief Inline file for arch.h.
 */

#include <string>
#include <algorithm>

#include <thrust/detail/util/blocking.h>
#include <thrust/system_error.h>

// #include this for make_uint3
#include <vector_functions.h>

namespace thrust
{
namespace experimental
{
namespace arch
{
namespace detail
{

inline void checked_get_current_device_properties(cudaDeviceProp &props)
{
  int current_device = -1;

  cudaError_t error = cudaGetDevice(&current_device);

  if(error)
  {
    throw thrust::experimental::system_error(error, thrust::experimental::cuda_category());
  }

  if(current_device < 0)
    throw thrust::experimental::system_error(thrust::experimental::cuda_errc::no_device, thrust::experimental::cuda_category());
  
  error = cudaGetDeviceProperties(&props, current_device);

  if(error)
  {
    throw thrust::experimental::system_error(error, thrust::experimental::cuda_category());
  }
} // end checked_get_current_device_properties()

template <typename KernelFunction>
void checked_get_function_attributes(cudaFuncAttributes& attributes, KernelFunction kernel)
{
  cudaError_t error = cudaFuncGetAttributes(&attributes, kernel);

  if(error)
  {
    throw thrust::experimental::system_error(error, thrust::experimental::cuda_category());
  }
} // end checked_get_function_attributes()

} // end detail


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


dim3 max_grid_dimensions(const cudaDeviceProp& properties)
{
    return make_uint3(properties.maxGridSize[0], properties.maxGridSize[1], properties.maxGridSize[2]);
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

dim3 max_grid_dimensions(void)
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

    size_t largest_blocksize  = std::min(properties.maxThreadsPerBlock, attributes.maxThreadsPerBlock);

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
    size_t largest_blocksize  = std::min(properties.maxThreadsPerBlock, attributes.maxThreadsPerBlock);

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

} // end namespace arch
} // end namespace experimental
} // end namespace thrust

