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

#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/type_traits.h>
#include <thrust/detail/backend/cuda/arch.h>

namespace thrust
{
namespace detail
{
namespace backend
{
namespace cuda
{
namespace detail
{

template <unsigned int _ThreadsPerBlock = 0,
          unsigned int _BlocksPerMultiprocessor = 0>
struct cuda_closure
{
  typedef thrust::detail::integral_constant<unsigned int, _ThreadsPerBlock>         ThreadsPerBlock;
  typedef thrust::detail::integral_constant<unsigned int, _BlocksPerMultiprocessor> BlocksPerMultiprocessor;

// CUDA built-in variables require nvcc
#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC
  __device__ __forceinline__ unsigned int thread_index(void)    { return threadIdx.x; }
  __device__ __forceinline__ unsigned int block_dimension(void) { return (_ThreadsPerBlock > 0) ? _ThreadsPerBlock : blockDim.x;  } // minor optimization
  __device__ __forceinline__ unsigned int block_index(void)     { return blockIdx.x;  }
  __device__ __forceinline__ unsigned int grid_dimension(void)  { return gridDim.x;   }
  __device__ __forceinline__ void         barrier(void)         { __syncthreads();    }
  __device__ __forceinline__ unsigned int linear_index(void)    { return block_dimension() * block_index() + thread_index(); }
#else
  __device__ __forceinline__ unsigned int thread_index(void)    { return 0; }
  __device__ __forceinline__ unsigned int block_dimension(void) { return 0; }
  __device__ __forceinline__ unsigned int block_index(void)     { return 0; }
  __device__ __forceinline__ unsigned int grid_dimension(void)  { return 0; }
  __device__ __forceinline__ void         barrier(void)         {           }
  __device__ __forceinline__ unsigned int linear_index(void)    { return 0; }
#endif // THRUST_DEVICE_COMPILER_NVCC
};

template<typename Closure, typename Size1, typename Size2>
  void launch_closure(Closure f, Size1 num_blocks, Size2 block_size);

template<typename Closure, typename Size1, typename Size2, typename Size3>
  void launch_closure(Closure f, Size1 num_blocks, Size2 block_size, Size3 smem_size);

/*! Returns a copy of the cudaFuncAttributes structure
 *  that is associated with a given Closure
 */
template <typename Closure>
arch::function_attributes_t closure_attributes(void);

} // end namespace detail
} // end namespace cuda
} // end namespace backend
} // end namespace detail
} // end namespace thrust

#include <thrust/detail/backend/cuda/detail/launch_closure.inl>

