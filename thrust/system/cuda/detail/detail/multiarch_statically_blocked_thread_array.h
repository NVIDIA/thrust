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

#include <thrust/detail/config.h>
#include <thrust/system/cuda/detail/detail/launch_closure.h>
#include <thrust/detail/static_map.h>


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


template<unsigned int default_block_size,
         unsigned int arch0,     unsigned int block_size0,
         unsigned int arch1 = 0, unsigned int block_size1 = default_block_size,
         unsigned int arch2 = 0, unsigned int block_size2 = default_block_size,
         unsigned int arch3 = 0, unsigned int block_size3 = default_block_size>
struct multiarch_statically_blocked_thread_array
{
  typedef thrust::detail::static_map<
    default_block_size,
    arch0, block_size0,
    arch1, block_size1,
    arch2, block_size2,
    arch3, block_size3
  > arch_to_block_size;


  typedef thrust::detail::integral_constant<
    unsigned int,
#ifdef __CUDA_ARCH__
    thrust::detail::static_lookup<
      __CUDA_ARCH__,
      arch_to_block_size
    >::value
#else
    0
#endif
  > ThreadsPerBlock;


  typedef thrust::detail::integral_constant<unsigned int,1> BlocksPerMultiprocessor;


  __device__ __thrust_forceinline__
  unsigned int thread_index() const
  {
#ifdef __CUDA_ARCH__
    const unsigned int cuda_arch = __CUDA_ARCH__;
#else
    const unsigned int cuda_arch = 0;
#endif

    const unsigned int block_size = thrust::detail::static_lookup<cuda_arch, arch_to_block_size>::value;

    using thrust::system::cuda::detail::detail::statically_blocked_thread_array;
    return statically_blocked_thread_array<block_size>().thread_index();
  }


  __device__ __thrust_forceinline__
  unsigned int block_dimension() const
  {
#ifdef __CUDA_ARCH__
   return thrust::detail::static_lookup<__CUDA_ARCH__, arch_to_block_size>::value;
#else
   return 0;
#endif
  }


  __host__ __device__ __thrust_forceinline__
  unsigned int block_dimension(unsigned int arch) const
  {
    return thrust::detail::lookup<arch_to_block_size>(arch);
  }


  __device__ __thrust_forceinline__
  unsigned int block_index() const
  {
#ifdef __CUDA_ARCH__
    const unsigned int cuda_arch = __CUDA_ARCH__;
#else
    const unsigned int cuda_arch = 0;
#endif

    const unsigned int block_size = thrust::detail::static_lookup<cuda_arch, arch_to_block_size>::value;

    using thrust::system::cuda::detail::detail::statically_blocked_thread_array;
    return statically_blocked_thread_array<block_size>().block_index();
  }


  __device__ __thrust_forceinline__
  unsigned int grid_dimension() const
  {
#ifdef __CUDA_ARCH__
    const unsigned int cuda_arch = __CUDA_ARCH__;
#else
    const unsigned int cuda_arch = 0;
#endif

    const unsigned int block_size = thrust::detail::static_lookup<cuda_arch, arch_to_block_size>::value;

    using thrust::system::cuda::detail::detail::statically_blocked_thread_array;
    return statically_blocked_thread_array<block_size>().grid_dimension();
  }


  __device__ __thrust_forceinline__
  unsigned int linear_index() const
  {
#if defined(__CUDA_ARCH__)
    const unsigned int cuda_arch = __CUDA_ARCH__;
#else
    const unsigned int cuda_arch = 0;
#endif

    const unsigned int block_size = thrust::detail::static_lookup<cuda_arch, arch_to_block_size>::value;

    using thrust::system::cuda::detail::detail::statically_blocked_thread_array;
    return statically_blocked_thread_array<block_size>().linear_index();
  }


  __device__ __thrust_forceinline__
  void barrier()
  {
#ifdef __CUDA_ARCH__
    const unsigned int cuda_arch = __CUDA_ARCH__;
#else
    const unsigned int cuda_arch = 0;
#endif

    const unsigned int block_size = thrust::detail::static_lookup<cuda_arch, arch_to_block_size>::value;

    using thrust::system::cuda::detail::detail::statically_blocked_thread_array;
    return statically_blocked_thread_array<block_size>().barrier();
  }
};


} // end detail
} // end detail
} // end cuda
} // end system
} // end thrust

