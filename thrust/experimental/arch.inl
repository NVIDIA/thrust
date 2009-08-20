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


/*! \file arch.inl
 *  \brief Inline file for arch.h.
 */

#include <cassert>
#include <string>

#include <thrust/experimental/arch.h>

#include <thrust/detail/util/blocking.h>

// #include this for make_uint3
#include <vector_functions.h>

// #include this for runtime_error
#include <stdexcept>

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
    throw std::runtime_error(std::string("CUDA error: ") + std::string(cudaGetErrorString(error)));
  } // end if

  if(current_device < 0)
  {
    throw std::runtime_error(std::string("No CUDA device found."));
  } // end if
  
  error = cudaGetDeviceProperties(&props, current_device);
  if(error)
  {
    throw std::runtime_error(std::string("CUDA error: ") + std::string(cudaGetErrorString(error)));
  } // end if
} // end checked_get_current_device_properties()

} // end detail


size_t num_multiprocessors(const cudaDeviceProp& properties)
{
    return properties.multiProcessorCount;
} // end num_multiprocessors()


size_t max_active_threads_per_multiprocessor(const cudaDeviceProp& properties)
{
    // index this array by [major, minor] revision
    // \see NVIDIA_CUDA_Programming_Guide_2.1.pdf pp 82--83
    static const size_t max_active_threads_by_compute_capability[2][4] = \
        {{     0,    0,    0,    0},
         {   768,  768, 1024, 1024}};

    assert(properties.major == 1);
    assert(properties.minor >= 0 && properties.minor <= 3);

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
    cudaError_t err = cudaFuncGetAttributes(&attributes, kernel);
    assert(err == cudaSuccess);

    return num_multiprocessors(properties) * max_active_blocks_per_multiprocessor(properties, attributes, CTA_SIZE, dynamic_smem_bytes);
}



} // end namespace arch

} // end namespace experimental

} // end namespace thrust

