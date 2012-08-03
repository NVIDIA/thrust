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

#pragma once

#include <cstddef>
#include <thrust/detail/config.h>

namespace thrust
{
namespace system
{
namespace cuda
{
namespace detail
{


// XXX define our own device_properties_t to avoid errors when #including
//     this file in the absence of a CUDA installation
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


// XXX define our own device_properties_t to avoid errors when #including
//     this file in the absence of a CUDA installation
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


/*! Computes a block size in number of threads for a CUDA kernel using a occupancy-promoting heuristic.
 *  \param attributes The cudaFuncAttributes corresponding to a __global__ function of interest on a GPU of interest.
 *  \param properties The cudaDeviceProp corresponding to a GPU on which to launch the __global__ function of interest.
 *  \return A CUDA block size, in number of threads, which the resources of the GPU's streaming multiprocessor can
 *          accomodate and which is intended to promote occupancy. The result is equivalent to the one performed by
 *          the "CUDA Occupancy Calculator". 
 *  \note The __global__ function of interest is presumed to use 0 bytes of dynamically-allocated __shared__ memory.
 */
inline __host__ __device__
std::size_t block_size_with_maximum_potential_occupancy(const function_attributes_t &attributes,
                                                        const device_properties_t   &properties);

/*! Computes a block size in number of threads for a CUDA kernel using a occupancy-promoting heuristic.
 *  Use this version of the function when a CUDA block's dynamically-allocated __shared__ memory requirements
 *  vary with the size of the block.
 *  \param attributes The cudaFuncAttributes corresponding to a __global__ function of interest on a GPU of interest.
 *  \param properties The cudaDeviceProp corresponding to a GPU on which to launch the __global__ function of interest.
 *  \param block_size_to_dynamic_smem_bytes A unary function which maps an integer CUDA block size to the number of bytes
 *         of dynamically-allocated __shared__ memory required by a CUDA block of that size.
 *  \return A CUDA block size, in number of threads, which the resources of the GPU's streaming multiprocessor can
 *          accomodate and which is intended to promote occupancy. The result is equivalent to the one performed by
 *          the "CUDA Occupancy Calculator". 
 */
template<typename UnaryFunction>
inline __host__ __device__
std::size_t block_size_with_maximum_potential_occupancy(const function_attributes_t &attributes,
                                                        const device_properties_t   &properties,
                                                        UnaryFunction block_size_to_dynamic_smem_size);


/*! Returns the maximum amount of dynamic shared memory each block
 *  can utilize without reducing thread occupancy.
 *
 *  \param properties CUDA device properties
 *  \param attributes CUDA function attributes
 *  \param blocks_per_processor Number of blocks per streaming multiprocessor
 */
inline __host__ __device__
size_t proportional_smem_allocation(const device_properties_t   &properties,
                                    const function_attributes_t &attributes,
                                    size_t blocks_per_processor);


template<typename UnaryFunction>
inline __host__ __device__
size_t max_blocksize_subject_to_smem_usage(const device_properties_t   &properties,
                                           const function_attributes_t &attributes,
                                           UnaryFunction blocksize_to_dynamic_smem_usage);



namespace cuda_launch_config_detail
{

using std::size_t;

namespace util
{


template<typename T>
inline __host__ __device__
T min_(const T &lhs, const T &rhs)
{
  return rhs < lhs ? rhs : lhs;
}


template <typename T>
struct zero_function
{
  inline __host__ __device__
  T operator()(T)
  {
    return 0;
  }
};


// x/y rounding towards +infinity for integers, used to determine # of blocks/warps etc.
template<typename L, typename R>
  inline __host__ __device__ L divide_ri(const L x, const R y)
{
    return (x + (y - 1)) / y;
}

// x/y rounding towards zero for integers, used to determine # of blocks/warps etc.
template<typename L, typename R>
  inline __host__ __device__ L divide_rz(const L x, const R y)
{
    return x / y;
}

// round x towards infinity to the next multiple of y
template<typename L, typename R>
  inline __host__ __device__ L round_i(const L x, const R y){ return y * divide_ri(x, y); }

// round x towards zero to the next multiple of y
template<typename L, typename R>
  inline __host__ __device__ L round_z(const L x, const R y){ return y * divide_rz(x, y); }

} // end namespace util



// granularity of shared memory allocation
inline __host__ __device__
size_t smem_allocation_unit(const device_properties_t &properties)
{
  switch(properties.major)
  {
    case 1:  return 512;
    case 2:  return 128;
    case 3:  return 256;
    default: return 256; // unknown GPU; have to guess
  }
}


// granularity of register allocation
inline __host__ __device__
size_t reg_allocation_unit(const device_properties_t &properties, const size_t regsPerThread)
{
  switch(properties.major)
  {
    case 1:  return (properties.minor <= 1) ? 256 : 512;
    case 2:  switch(regsPerThread)
             {
               case 21:
               case 22:
               case 29:
               case 30:
               case 37:
               case 38:
               case 45:
               case 46:
                 return 128;
               default:
                 return 64;
             }
    case 3:  return 256;
    default: return 256; // unknown GPU; have to guess
  }
}


// granularity of warp allocation
inline __host__ __device__
size_t warp_allocation_multiple(const device_properties_t &properties)
{
  return (properties.major <= 1) ? 2 : 1;
}

// number of "sides" into which the multiprocessor is partitioned
inline __host__ __device__
size_t num_sides_per_multiprocessor(const device_properties_t &properties)
{
  switch(properties.major)
  {
    case 1:  return 1;
    case 2:  return 2;
    case 3:  return 4;
    default: return 4; // unknown GPU; have to guess
  }
}


inline __host__ __device__
size_t max_blocks_per_multiprocessor(const device_properties_t &properties)
{
  return (properties.major <= 2) ? 8 : 16;
}


inline __host__ __device__
size_t max_active_blocks_per_multiprocessor(const device_properties_t    &properties,
                                            const function_attributes_t  &attributes,
                                            int CTA_SIZE,
                                            size_t dynamic_smem_bytes)
{
  // Determine the maximum number of CTAs that can be run simultaneously per SM
  // This is equivalent to the calculation done in the CUDA Occupancy Calculator spreadsheet

  //////////////////////////////////////////
  // Limits due to threads/SM or blocks/SM
  //////////////////////////////////////////
  const size_t maxThreadsPerSM = properties.maxThreadsPerMultiProcessor;  // 768, 1024, 1536, etc.
  const size_t maxBlocksPerSM  = max_blocks_per_multiprocessor(properties);

  // Calc limits
  const size_t ctaLimitThreads = (CTA_SIZE <= properties.maxThreadsPerBlock) ? maxThreadsPerSM / CTA_SIZE : 0;
  const size_t ctaLimitBlocks  = maxBlocksPerSM;

  //////////////////////////////////////////
  // Limits due to shared memory/SM
  //////////////////////////////////////////
  const size_t smemAllocationUnit     = smem_allocation_unit(properties);
  const size_t smemBytes  = attributes.sharedSizeBytes + dynamic_smem_bytes;
  const size_t smemPerCTA = util::round_i(smemBytes, smemAllocationUnit);

  // Calc limit
  const size_t ctaLimitSMem = smemPerCTA > 0 ? properties.sharedMemPerBlock / smemPerCTA : maxBlocksPerSM;

  //////////////////////////////////////////
  // Limits due to registers/SM
  //////////////////////////////////////////
  const size_t regAllocationUnit      = reg_allocation_unit(properties, attributes.numRegs);
  const size_t warpAllocationMultiple = warp_allocation_multiple(properties);
  const size_t numWarps = util::round_i(util::divide_ri(CTA_SIZE, properties.warpSize), warpAllocationMultiple);

  // Calc limit
  size_t ctaLimitRegs;
  if(properties.major <= 1)
  {
    // GPUs of compute capability 1.x allocate registers to CTAs
    // Number of regs per block is regs per thread times number of warps times warp size, rounded up to allocation unit
    const size_t regsPerCTA = util::round_i(attributes.numRegs * properties.warpSize * numWarps, regAllocationUnit);
    ctaLimitRegs = regsPerCTA > 0 ? properties.regsPerBlock / regsPerCTA : maxBlocksPerSM;
  }
  else
  {
    // GPUs of compute capability 2.x and higher allocate registers to warps
    // Number of regs per warp is regs per thread times times warp size, rounded up to allocation unit
    const size_t regsPerWarp = util::round_i(attributes.numRegs * properties.warpSize, regAllocationUnit);
    const size_t numSides = num_sides_per_multiprocessor(properties);
    const size_t numRegsPerSide = properties.regsPerBlock / numSides;
    ctaLimitRegs = regsPerWarp > 0 ? ((numRegsPerSide / regsPerWarp) * numSides) / numWarps : maxBlocksPerSM;
  }

  //////////////////////////////////////////
  // Overall limit is min() of limits due to above reasons
  //////////////////////////////////////////
  return util::min_(ctaLimitRegs, util::min_(ctaLimitSMem, util::min_(ctaLimitThreads, ctaLimitBlocks)));
}


} // end namespace cuda_launch_config_detail


template<typename UnaryFunction>
inline __host__ __device__
std::size_t block_size_with_maximum_potential_occupancy(const function_attributes_t &attributes,
                                                        const device_properties_t   &properties,
                                                        UnaryFunction block_size_to_dynamic_smem_size)
{
  size_t max_occupancy      = properties.maxThreadsPerMultiProcessor;
  size_t largest_blocksize  = cuda_launch_config_detail::util::min_(properties.maxThreadsPerBlock, attributes.maxThreadsPerBlock);
  size_t granularity        = properties.warpSize;
  size_t max_blocksize      = 0;
  size_t highest_occupancy  = 0;

  for(size_t blocksize = largest_blocksize; blocksize != 0; blocksize -= granularity)
  {
    size_t occupancy = blocksize * cuda_launch_config_detail::max_active_blocks_per_multiprocessor(properties, attributes, blocksize, block_size_to_dynamic_smem_size(blocksize));

    if(occupancy > highest_occupancy)
    {
      max_blocksize = blocksize;
      highest_occupancy = occupancy;
    }

    // early out, can't do better
    if(highest_occupancy == max_occupancy)
      break;
  }

  return max_blocksize;
}


inline __host__ __device__
std::size_t block_size_with_maximum_potential_occupancy(const function_attributes_t &attributes,
                                                        const device_properties_t   &properties)
{
  return block_size_with_maximum_potential_occupancy(attributes, properties, cuda_launch_config_detail::util::zero_function<std::size_t>());
}


inline __host__ __device__
size_t proportional_smem_allocation(const device_properties_t   &properties,
                                    const function_attributes_t &attributes,
                                    size_t blocks_per_processor)
{
  size_t smem_per_processor    = properties.sharedMemPerBlock;
  size_t smem_allocation_unit  = cuda_launch_config_detail::smem_allocation_unit(properties);

  size_t total_smem_per_block  = cuda_launch_config_detail::util::round_z(smem_per_processor / blocks_per_processor, smem_allocation_unit);
  size_t static_smem_per_block = attributes.sharedSizeBytes;
  
  return total_smem_per_block - static_smem_per_block;
}


template<typename UnaryFunction>
inline __host__ __device__
size_t max_blocksize_subject_to_smem_usage(const device_properties_t   &properties,
                                           const function_attributes_t &attributes,
                                           UnaryFunction blocksize_to_dynamic_smem_usage)
{
  size_t largest_blocksize = (thrust::min)(properties.maxThreadsPerBlock, attributes.maxThreadsPerBlock);
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


} // end detail
} // end cuda
} // end system
} // end thrust

