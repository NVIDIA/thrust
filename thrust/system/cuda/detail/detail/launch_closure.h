/*
 *  Copyright 2008-2013 NVIDIA Corporation
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
#include <thrust/system/cuda/detail/cuda_launch_config.h>
#include <thrust/system/cuda/detail/execution_policy.h>

namespace thrust
{
namespace system
{
namespace cuda
{
namespace detail
{
namespace detail
{


template<unsigned int _ThreadsPerBlock = 0,
         unsigned int _BlocksPerMultiprocessor = 0>
struct launch_bounds
{
  typedef thrust::detail::integral_constant<unsigned int, _ThreadsPerBlock>         ThreadsPerBlock;
  typedef thrust::detail::integral_constant<unsigned int, _BlocksPerMultiprocessor> BlocksPerMultiprocessor;
};


struct thread_array : public launch_bounds<>
{
// CUDA built-in variables require nvcc
#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC
  __device__ __thrust_forceinline__ unsigned int thread_index(void) const { return threadIdx.x; }
  __device__ __thrust_forceinline__ unsigned int thread_count(void) const { return blockDim.x * gridDim.x; } 
#else
  __device__ __thrust_forceinline__ unsigned int thread_index(void) const { return 0; }
  __device__ __thrust_forceinline__ unsigned int thread_count(void) const { return 0; } 
#endif // THRUST_DEVICE_COMPILER_NVCC
};


struct blocked_thread_array : public launch_bounds<>
{
// CUDA built-in variables require nvcc
#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC
  __device__ __thrust_forceinline__ unsigned int thread_index(void)    const { return threadIdx.x; }
  __device__ __thrust_forceinline__ unsigned int block_dimension(void) const { return blockDim.x;  } 
  __device__ __thrust_forceinline__ unsigned int block_index(void)     const { return blockIdx.x;  }
  __device__ __thrust_forceinline__ unsigned int grid_dimension(void)  const { return gridDim.x;   }
  __device__ __thrust_forceinline__ unsigned int linear_index(void)    const { return block_dimension() * block_index() + thread_index(); }
  __device__ __thrust_forceinline__ void         barrier(void)               { __syncthreads();    }
#else
  __device__ __thrust_forceinline__ unsigned int thread_index(void)    const { return 0; }
  __device__ __thrust_forceinline__ unsigned int block_dimension(void) const { return 0; }
  __device__ __thrust_forceinline__ unsigned int block_index(void)     const { return 0; }
  __device__ __thrust_forceinline__ unsigned int grid_dimension(void)  const { return 0; }
  __device__ __thrust_forceinline__ unsigned int linear_index(void)    const { return 0; }
  __device__ __thrust_forceinline__ void         barrier(void)               {           }
#endif // THRUST_DEVICE_COMPILER_NVCC
};


template <unsigned int _ThreadsPerBlock>
struct statically_blocked_thread_array : public launch_bounds<_ThreadsPerBlock,1>
{
// CUDA built-in variables require nvcc
#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC
  __device__ __thrust_forceinline__ unsigned int thread_index(void)    const { return threadIdx.x;      }
  __device__ __thrust_forceinline__ unsigned int block_dimension(void) const { return _ThreadsPerBlock; } // minor optimization
  __device__ __thrust_forceinline__ unsigned int block_index(void)     const { return blockIdx.x;       }
  __device__ __thrust_forceinline__ unsigned int grid_dimension(void)  const { return gridDim.x;        }
  __device__ __thrust_forceinline__ unsigned int linear_index(void)    const { return block_dimension() * block_index() + thread_index(); }
  __device__ __thrust_forceinline__ void         barrier(void)               { __syncthreads();    }
#else
  __device__ __thrust_forceinline__ unsigned int thread_index(void)    const { return 0; }
  __device__ __thrust_forceinline__ unsigned int block_dimension(void) const { return 0; }
  __device__ __thrust_forceinline__ unsigned int block_index(void)     const { return 0; }
  __device__ __thrust_forceinline__ unsigned int grid_dimension(void)  const { return 0; }
  __device__ __thrust_forceinline__ unsigned int linear_index(void)    const { return 0; }
  __device__ __thrust_forceinline__ void         barrier(void)               {           }
#endif // THRUST_DEVICE_COMPILER_NVCC
};

template<typename DerivedPolicy, typename Closure, typename Size>
__host__ __device__
void launch_closure(execution_policy<DerivedPolicy> &exec, Closure f, Size num_blocks);

template<typename DerivedPolicy, typename Closure, typename Size1, typename Size2>
__host__ __device__
void launch_closure(execution_policy<DerivedPolicy> &exec, Closure f, Size1 num_blocks, Size2 block_size);

template<typename DerivedPolicy, typename Closure, typename Size1, typename Size2, typename Size3>
__host__ __device__
void launch_closure(execution_policy<DerivedPolicy> &exec, Closure f, Size1 num_blocks, Size2 block_size, Size3 smem_size);

/*! Returns a copy of the cudaFuncAttributes structure
 *  that is associated with a given Closure
 */
template<typename Closure>
__host__ __device__
function_attributes_t closure_attributes(void);


} // end namespace detail
} // end namespace detail
} // end namespace cuda
} // end namespace system
} // end namespace thrust

#include <thrust/system/cuda/detail/detail/launch_closure.inl>

