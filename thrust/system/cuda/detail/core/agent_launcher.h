/******************************************************************************
 * Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/
#pragma once

#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC
#include <thrust/detail/config.h>
#include <thrust/system/cuda/detail/guarded_cuda_runtime_api.h>
#include <thrust/system/cuda/detail/core/triple_chevron_launch.h>
#include <thrust/system/cuda/detail/core/util.h>
#include <cassert>

BEGIN_NS_THRUST
namespace cuda_cub {
namespace core {


#ifdef __CUDA_ARCH__
#if 0
  template <class Agent, class... Args>
  void __global__ 
  __launch_bounds__(Agent::ptx_plan::BLOCK_THREADS,Agent::ptx_plan::MIN_BLOCKS)
      _kernel_agent(Args... args)
  {
    extern __shared__ char shmem[];
    Agent::entry(args..., shmem);
  }
#else
  template <class Agent, class _0>
  void __global__ __launch_bounds__(Agent::ptx_plan::BLOCK_THREADS, Agent::ptx_plan::MIN_BLOCKS)
  _kernel_agent(_0 x0)
  {
    extern __shared__ char shmem[];
    Agent::entry(x0, shmem);
  }
  template <class Agent, class _0, class _1>
  void __global__ __launch_bounds__(Agent::ptx_plan::BLOCK_THREADS, Agent::ptx_plan::MIN_BLOCKS)
  _kernel_agent(_0 x0, _1 x1)
  {
    extern __shared__ char shmem[];
    Agent::entry(x0, x1, shmem);
  }
  template <class Agent, class _0, class _1, class _2>
  void __global__ __launch_bounds__(Agent::ptx_plan::BLOCK_THREADS, Agent::ptx_plan::MIN_BLOCKS)
  _kernel_agent(_0 x0, _1 x1, _2 x2)
  {
    extern __shared__ char shmem[];
    Agent::entry(x0, x1, x2, shmem);
  }
  template <class Agent, class _0, class _1, class _2, class _3>
  void __global__ __launch_bounds__(Agent::ptx_plan::BLOCK_THREADS, Agent::ptx_plan::MIN_BLOCKS)
  _kernel_agent(_0 x0, _1 x1, _2 x2, _3 x3)
  {
    extern __shared__ char shmem[];
    Agent::entry(x0, x1, x2, x3, shmem);
  }
  template <class Agent, class _0, class _1, class _2, class _3, class _4>
  void __global__ __launch_bounds__(Agent::ptx_plan::BLOCK_THREADS, Agent::ptx_plan::MIN_BLOCKS)
  _kernel_agent(_0 x0, _1 x1, _2 x2, _3 x3, _4 x4)
  {
    extern __shared__ char shmem[];
    Agent::entry(x0, x1, x2, x3, x4, shmem);
  }
  template <class Agent, class _0, class _1, class _2, class _3, class _4, class _5>
  void __global__ __launch_bounds__(Agent::ptx_plan::BLOCK_THREADS, Agent::ptx_plan::MIN_BLOCKS)
  _kernel_agent(_0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5)
  {
    extern __shared__ char shmem[];
    Agent::entry(x0, x1, x2, x3, x4, x5, shmem);
  }
  template <class Agent, class _0, class _1, class _2, class _3, class _4, class _5, class _6>
  void __global__ __launch_bounds__(Agent::ptx_plan::BLOCK_THREADS, Agent::ptx_plan::MIN_BLOCKS)
  _kernel_agent(_0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6)
  {
    extern __shared__ char shmem[];
    Agent::entry(x0, x1, x2, x3, x4, x5, x6, shmem);
  }
  template <class Agent, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7>
  void __global__ __launch_bounds__(Agent::ptx_plan::BLOCK_THREADS, Agent::ptx_plan::MIN_BLOCKS)
  _kernel_agent(_0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7)
  {
    extern __shared__ char shmem[];
    Agent::entry(x0, x1, x2, x3, x4, x5, x6, x7, shmem);
  }
  template <class Agent, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8>
  void __global__ __launch_bounds__(Agent::ptx_plan::BLOCK_THREADS, Agent::ptx_plan::MIN_BLOCKS)
  _kernel_agent(_0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8)
  {
    extern __shared__ char shmem[];
    Agent::entry(x0, x1, x2, x3, x4, x5, x6, x7, x8, shmem);
  }
  template <class Agent, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9>
  void __global__ __launch_bounds__(Agent::ptx_plan::BLOCK_THREADS, Agent::ptx_plan::MIN_BLOCKS)
  _kernel_agent(_0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8, _9 x9)
  {
    extern __shared__ char shmem[];
    Agent::entry(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, shmem);
  }
  template <class Agent, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _A>
  void __global__ __launch_bounds__(Agent::ptx_plan::BLOCK_THREADS, Agent::ptx_plan::MIN_BLOCKS)
  _kernel_agent(_0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8, _9 x9, _A xA)
  {
    extern __shared__ char shmem[];
    Agent::entry(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, xA, shmem);
  }
  template <class Agent, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _A, class _B>
  void __global__ __launch_bounds__(Agent::ptx_plan::BLOCK_THREADS, Agent::ptx_plan::MIN_BLOCKS)
  _kernel_agent(_0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8, _9 x9, _A xA, _B xB)
  {
    extern __shared__ char shmem[];
    Agent::entry(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, xA, xB, shmem);
  }
  template <class Agent, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _A, class _B, class _C>
  void __global__ __launch_bounds__(Agent::ptx_plan::BLOCK_THREADS, Agent::ptx_plan::MIN_BLOCKS)
  _kernel_agent(_0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8, _9 x9, _A xA, _B xB, _C xC)
  {
    extern __shared__ char shmem[];
    Agent::entry(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, xA, xB, xC, shmem);
  }
  template <class Agent, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _A, class _B, class _C, class _D>
  void __global__ __launch_bounds__(Agent::ptx_plan::BLOCK_THREADS, Agent::ptx_plan::MIN_BLOCKS)
  _kernel_agent(_0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8, _9 x9, _A xA, _B xB, _C xC, _D xD)
  {
    extern __shared__ char shmem[];
    Agent::entry(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, xA, xB, xC, xD, shmem);
  }
  template <class Agent, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _A, class _B, class _C, class _D, class _E>
  void __global__ __launch_bounds__(Agent::ptx_plan::BLOCK_THREADS, Agent::ptx_plan::MIN_BLOCKS)
  _kernel_agent(_0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8, _9 x9, _A xA, _B xB, _C xC, _D xD, _E xE)
  {
    extern __shared__ char shmem[];
    Agent::entry(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, xA, xB, xC, xD, xE, shmem);
  }
#endif
  
  ////////////////////////////////////////////////////////////


#if 0
  template <class Agent, class... Args>
  void __global__ 
  __launch_bounds__(Agent::ptx_plan::BLOCK_THREADS,Agent::ptx_plan::MIN_BLOCKS)
      _kernel_agent_vshmem(char* vshmem, Args... args)
  {
    vshmem += blockIdx.x * temp_storage_size<typename Agent::ptx_plan>::value;
    Agent::entry(args..., vshmem);
  }
#else
  template <class Agent, class _0>
  void __global__ __launch_bounds__(Agent::ptx_plan::BLOCK_THREADS, Agent::ptx_plan::MIN_BLOCKS)
  _kernel_agent_vshmem(char* vshmem, _0 x0)
  {
    vshmem += blockIdx.x * temp_storage_size<typename Agent::ptx_plan>::value;
    Agent::entry(x0, vshmem);
  }
  template <class Agent, class _0, class _1>
  void __global__ __launch_bounds__(Agent::ptx_plan::BLOCK_THREADS, Agent::ptx_plan::MIN_BLOCKS)
  _kernel_agent_vshmem(char* vshmem, _0 x0, _1 x1)
  {
    vshmem += blockIdx.x * temp_storage_size<typename Agent::ptx_plan>::value;
    Agent::entry(x0, x1, vshmem);
  }
  template <class Agent, class _0, class _1, class _2>
  void __global__ __launch_bounds__(Agent::ptx_plan::BLOCK_THREADS, Agent::ptx_plan::MIN_BLOCKS)
  _kernel_agent_vshmem(char* vshmem, _0 x0, _1 x1, _2 x2)
  {
    vshmem += blockIdx.x * temp_storage_size<typename Agent::ptx_plan>::value;
    Agent::entry(x0, x1, x2, vshmem);
  }
  template <class Agent, class _0, class _1, class _2, class _3>
  void __global__ __launch_bounds__(Agent::ptx_plan::BLOCK_THREADS, Agent::ptx_plan::MIN_BLOCKS)
  _kernel_agent_vshmem(char* vshmem, _0 x0, _1 x1, _2 x2, _3 x3)
  {
    vshmem += blockIdx.x * temp_storage_size<typename Agent::ptx_plan>::value;
    Agent::entry(x0, x1, x2, x3, vshmem);
  }
  template <class Agent, class _0, class _1, class _2, class _3, class _4>
  void __global__ __launch_bounds__(Agent::ptx_plan::BLOCK_THREADS, Agent::ptx_plan::MIN_BLOCKS)
  _kernel_agent_vshmem(char* vshmem, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4)
  {
    vshmem += blockIdx.x * temp_storage_size<typename Agent::ptx_plan>::value;
    Agent::entry(x0, x1, x2, x3, x4, vshmem);
  }
  template <class Agent, class _0, class _1, class _2, class _3, class _4, class _5>
  void __global__ __launch_bounds__(Agent::ptx_plan::BLOCK_THREADS, Agent::ptx_plan::MIN_BLOCKS)
  _kernel_agent_vshmem(char* vshmem, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5)
  {
    vshmem += blockIdx.x * temp_storage_size<typename Agent::ptx_plan>::value;
    Agent::entry(x0, x1, x2, x3, x4, x5, vshmem);
  }
  template <class Agent, class _0, class _1, class _2, class _3, class _4, class _5, class _6>
  void __global__ __launch_bounds__(Agent::ptx_plan::BLOCK_THREADS, Agent::ptx_plan::MIN_BLOCKS)
  _kernel_agent_vshmem(char* vshmem, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6)
  {
    vshmem += blockIdx.x * temp_storage_size<typename Agent::ptx_plan>::value;
    Agent::entry(x0, x1, x2, x3, x4, x5, x6, vshmem);
  }
  template <class Agent, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7>
  void __global__ __launch_bounds__(Agent::ptx_plan::BLOCK_THREADS, Agent::ptx_plan::MIN_BLOCKS)
  _kernel_agent_vshmem(char* vshmem, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7)
  {
    vshmem += blockIdx.x * temp_storage_size<typename Agent::ptx_plan>::value;
    Agent::entry(x0, x1, x2, x3, x4, x5, x6, x7, vshmem);
  }
  template <class Agent, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8>
  void __global__ __launch_bounds__(Agent::ptx_plan::BLOCK_THREADS, Agent::ptx_plan::MIN_BLOCKS)
  _kernel_agent_vshmem(char* vshmem, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8)
  {
    vshmem += blockIdx.x * temp_storage_size<typename Agent::ptx_plan>::value;
    Agent::entry(x0, x1, x2, x3, x4, x5, x6, x7, x8, vshmem);
  }
  template <class Agent, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9>
  void __global__ __launch_bounds__(Agent::ptx_plan::BLOCK_THREADS, Agent::ptx_plan::MIN_BLOCKS)
  _kernel_agent_vshmem(char* vshmem, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8, _9 x9)
  {
    vshmem += blockIdx.x * temp_storage_size<typename Agent::ptx_plan>::value;
    Agent::entry(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, vshmem);
  }
  template <class Agent, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _A>
  void __global__ __launch_bounds__(Agent::ptx_plan::BLOCK_THREADS, Agent::ptx_plan::MIN_BLOCKS)
  _kernel_agent_vshmem(char* vshmem, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8, _9 x9, _A xA)
  {
    vshmem += blockIdx.x * temp_storage_size<typename Agent::ptx_plan>::value;
    Agent::entry(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, xA, vshmem);
  }
  template <class Agent, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _A, class _B>
  void __global__ __launch_bounds__(Agent::ptx_plan::BLOCK_THREADS, Agent::ptx_plan::MIN_BLOCKS)
  _kernel_agent_vshmem(char* vshmem, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8, _9 x9, _A xA, _B xB)
  {
    vshmem += blockIdx.x * temp_storage_size<typename Agent::ptx_plan>::value;
    Agent::entry(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, xA, xB, vshmem);
  }
  template <class Agent, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _A, class _B, class _C>
  void __global__ __launch_bounds__(Agent::ptx_plan::BLOCK_THREADS, Agent::ptx_plan::MIN_BLOCKS)
  _kernel_agent_vshmem(char* vshmem, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8, _9 x9, _A xA, _B xB, _C xC)
  {
    vshmem += blockIdx.x * temp_storage_size<typename Agent::ptx_plan>::value;
    Agent::entry(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, xA, xB, xC, vshmem);
  }
  template <class Agent, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _A, class _B, class _C, class _D>
  void __global__ __launch_bounds__(Agent::ptx_plan::BLOCK_THREADS, Agent::ptx_plan::MIN_BLOCKS)
  _kernel_agent_vshmem(char* vshmem, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8, _9 x9, _A xA, _B xB, _C xC, _D xD)
  {
    vshmem += blockIdx.x * temp_storage_size<typename Agent::ptx_plan>::value;
    Agent::entry(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, xA, xB, xC, xD, vshmem);
  }
  template <class Agent, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _A, class _B, class _C, class _D, class _E>
  void __global__ __launch_bounds__(Agent::ptx_plan::BLOCK_THREADS, Agent::ptx_plan::MIN_BLOCKS)
  _kernel_agent_vshmem(char* vshmem, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8, _9 x9, _A xA, _B xB, _C xC, _D xD, _E xE)
  {
    vshmem += blockIdx.x * temp_storage_size<typename Agent::ptx_plan>::value;
    Agent::entry(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, xA, xB, xC, xD, xE, vshmem);
  }
#endif
#else
#if 0
  template <class , class... Args >
  void __global__  _kernel_agent(Args... args) {}
  template <class , class... Args >
  void __global__  _kernel_agent_vshmem(char*, Args... args) {}
#else
  template <class, class _0>
  void __global__ _kernel_agent(_0) {}
  template <class, class _0, class _1>
  void __global__ _kernel_agent(_0,_1) {}
  template <class, class _0, class _1, class _2>
  void __global__ _kernel_agent(_0,_1,_2) {}
  template <class, class _0, class _1, class _2, class _3>
  void __global__ _kernel_agent(_0,_1,_2,_3) {}
  template <class, class _0, class _1, class _2, class _3, class _4>
  void __global__ _kernel_agent(_0,_1,_2,_3, _4) {}
  template <class, class _0, class _1, class _2, class _3, class _4, class _5>
  void __global__ _kernel_agent(_0,_1,_2,_3, _4, _5) {}
  template <class, class _0, class _1, class _2, class _3, class _4, class _5, class _6>
  void __global__ _kernel_agent(_0,_1,_2,_3, _4, _5, _6) {}
  template <class, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7>
  void __global__ _kernel_agent(_0,_1,_2,_3, _4, _5, _6, _7) {}
  template <class, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8>
  void __global__ _kernel_agent(_0,_1,_2,_3, _4, _5, _6, _7, _8) {}
  template <class, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9>
  void __global__ _kernel_agent(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9) {}
  template <class, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _A>
  void __global__ _kernel_agent(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _A) {}
  template <class, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _A, class _B>
  void __global__ _kernel_agent(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _A, _B) {}
  template <class, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _A, class _B, class _C>
  void __global__ _kernel_agent(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _A, _B,_C) {}
  template <class, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _A, class _B, class _C, class _D>
  void __global__ _kernel_agent(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _A, _B,_C, _D) {}
  template <class, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _A, class _B, class _C, class _D, class _E>
  void __global__ _kernel_agent(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _A, _B,_C, _D, _E) {}
  ////////////////////////////////////////////////////////////
  template <class, class _0>
  void __global__ _kernel_agent_vshmem(char*,_0) {}
  template <class, class _0, class _1>
  void __global__ _kernel_agent_vshmem(char*,_0,_1) {}
  template <class, class _0, class _1, class _2>
  void __global__ _kernel_agent_vshmem(char*,_0,_1,_2) {}
  template <class, class _0, class _1, class _2, class _3>
  void __global__ _kernel_agent_vshmem(char*,_0,_1,_2,_3) {}
  template <class, class _0, class _1, class _2, class _3, class _4>
  void __global__ _kernel_agent_vshmem(char*,_0,_1,_2,_3, _4) {}
  template <class, class _0, class _1, class _2, class _3, class _4, class _5>
  void __global__ _kernel_agent_vshmem(char*,_0,_1,_2,_3, _4, _5) {}
  template <class, class _0, class _1, class _2, class _3, class _4, class _5, class _6>
  void __global__ _kernel_agent_vshmem(char*,_0,_1,_2,_3, _4, _5, _6) {}
  template <class, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7>
  void __global__ _kernel_agent_vshmem(char*,_0,_1,_2,_3, _4, _5, _6, _7) {}
  template <class, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8>
  void __global__ _kernel_agent_vshmem(char*,_0,_1,_2,_3, _4, _5, _6, _7, _8) {}
  template <class, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9>
  void __global__ _kernel_agent_vshmem(char*,_0, _1, _2, _3, _4, _5, _6, _7, _8, _9) {}
  template <class, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _A>
  void __global__ _kernel_agent_vshmem(char*,_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _A) {}
  template <class, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _A, class _B>
  void __global__ _kernel_agent_vshmem(char*,_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _A, _B) {}
  template <class, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _A, class _B, class _C>
  void __global__ _kernel_agent_vshmem(char*,_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _A, _B, _C) {}
  template <class, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _A, class _B, class _C, class _D>
  void __global__ _kernel_agent_vshmem(char*,_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _A, _B, _C, _D) {}
  template <class, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _A, class _B, class _C, class _D, class _E>
  void __global__ _kernel_agent_vshmem(char*,_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _A, _B, _C, _D, _E) {}
#endif
#endif


  template<class Agent>
  struct AgentLauncher : Agent
  {
    core::AgentPlan plan;
    size_t          count;
    cudaStream_t    stream;
    char const*     name;
    bool            debug_sync;
    unsigned int    grid;
    char*           vshmem;
    bool            has_shmem;

    enum
    {
      MAX_SHMEM_PER_BLOCK = 48 * 1024,
    };
    typedef
        typename has_enough_shmem<Agent,
                                  MAX_SHMEM_PER_BLOCK>::type has_enough_shmem_t;

    template <class Size>
    CUB_RUNTIME_FUNCTION
    AgentLauncher(AgentPlan    plan_,
                  Size         count_,
                  cudaStream_t stream_,
                  char const*  name_,
                  bool         debug_sync_)
        : plan(plan_),
          count((size_t)count_),
          stream(stream_),
          name(name_),
          debug_sync(debug_sync_),
          grid((count + plan.items_per_tile - 1) / plan.items_per_tile),
          vshmem(NULL),
          has_shmem((size_t)core::get_max_shared_memory_per_block() >= (size_t)plan.shared_memory_size)
    {
      assert(count > 0);
    }

    template <class Size>
    CUB_RUNTIME_FUNCTION
    AgentLauncher(AgentPlan    plan_,
                  Size         count_,
                  cudaStream_t stream_,
                  char*        vshmem,
                  char const*  name_,
                  bool         debug_sync_)
        : plan(plan_),
          count((size_t)count_),
          stream(stream_),
          name(name_),
          debug_sync(debug_sync_),
          grid((count + plan.items_per_tile - 1) / plan.items_per_tile),
          vshmem(vshmem),
          has_shmem((size_t)core::get_max_shared_memory_per_block() >= (size_t)plan.shared_memory_size)
    {
      assert(count > 0);
    }
    
    CUB_RUNTIME_FUNCTION
    AgentLauncher(AgentPlan    plan_,
                  cudaStream_t stream_,
                  char const*  name_,
                  bool         debug_sync_)
        : plan(plan_),
          count(0),
          stream(stream_),
          name(name_),
          debug_sync(debug_sync_),
          grid(plan.grid_size),
          vshmem(NULL),
          has_shmem((size_t)core::get_max_shared_memory_per_block() >= (size_t)plan.shared_memory_size)
    {
      assert(plan.grid_size > 0);
    }

    CUB_RUNTIME_FUNCTION
    AgentLauncher(AgentPlan    plan_,
                  cudaStream_t stream_,
                  char*        vshmem,
                  char const*  name_,
                  bool         debug_sync_)
        : plan(plan_),
          count(0),
          stream(stream_),
          name(name_),
          debug_sync(debug_sync_),
          grid(plan.grid_size),
          vshmem(vshmem),
          has_shmem((size_t)core::get_max_shared_memory_per_block() >= (size_t)plan.shared_memory_size)
    {
      assert(plan.grid_size > 0);
    }

#if 0
    THRUST_RUNTIME_FUNCTION
    AgentPlan static get_plan(cudaStream_t s, void* d_ptr = 0)
    {
      // in separable compilation mode, we have no choice
      // but to call kernel to get agent_plan
      // otherwise the risk is something may fail
      // if user mix & match ptx versions in a separably compiled function
      // http://nvbugs/1772071
      // XXX may be it is too string of a requirements, consider relaxing it in
      // the future
#ifdef __CUDACC_RDC__
      return core::get_agent_plan<Agent>(s, d_ptr);
#else
      core::cuda_optional<int> ptx_version = core::get_ptx_version();
      //CUDA_CUB_RET_IF_FAIL(ptx_version.status());
      return get_agent_plan<Agent>(ptx_version);
#endif
    }
    THRUST_RUNTIME_FUNCTION
    AgentPlan static get_plan_default()
    {
      return get_agent_plan<Agent>(sm_arch<0>::type::ver);
    }
#endif
    
    CUB_RUNTIME_FUNCTION
    typename core::get_plan<Agent>::type static get_plan(cudaStream_t s, void* d_ptr = 0)
    {
      core::cuda_optional<int> ptx_version = core::get_ptx_version();
      return get_agent_plan<Agent>(ptx_version);
    }
    
    THRUST_RUNTIME_FUNCTION
    typename core::get_plan<Agent>::type static get_plan()
    {
      return get_agent_plan<Agent>(sm_arch<0>::type::ver);
    }

    CUB_RUNTIME_FUNCTION void sync() const
    {
      if (debug_sync)
      {
#ifdef __CUDA_ARCH__
        cudaDeviceSynchronize();
#else
        cudaStreamSynchronize(stream);
#endif
      }
    }

    template<class K>
    static cuda_optional<int> THRUST_RUNTIME_FUNCTION
    max_blocks_per_sm_impl(K k, int block_threads)
    {
      int occ;
      cudaError_t status = cub::MaxSmOccupancy(occ, k, block_threads);
      return cuda_optional<int>(status == cudaSuccess ? occ : -1, status);
    }

    template <class K>
    cuda_optional<int> THRUST_RUNTIME_FUNCTION
    max_sm_occupancy(K k) const
    {
      return max_blocks_per_sm_impl(k, plan.block_threads);
    }


    
    template<class K>
    THRUST_RUNTIME_FUNCTION
    void print_info(K k) const
    {
      if (debug_sync)
      {
        cuda_optional<int> occ = max_sm_occupancy(k);
        core::cuda_optional<int> ptx_version = core::get_ptx_version();
        if (count > 0)
        {
          _CubLog("Invoking %s<<<%u, %d, %d, %lld>>>(), %llu items total, %d items per thread, %d SM occupancy, %d vshmem size, %d ptx_version \n",
                  name,
                  grid,
                  plan.block_threads,
                  (has_shmem ? (int)plan.shared_memory_size : 0),
                  (long long)stream,
                  (long long)count,
                  plan.items_per_thread,
                  (int)occ,
                  (!has_shmem ? (int)plan.shared_memory_size : 0),
                  (int)ptx_version);
        }
        else
        {
          _CubLog("Invoking %s<<<%u, %d, %d, %lld>>>(), %d items per thread, %d SM occupancy, %d vshmem size, %d ptx_version\n",
                  name,
                  grid,
                  plan.block_threads,
                  (has_shmem ? (int)plan.shared_memory_size : 0),
                  (long long)stream,
                  plan.items_per_thread,
                  (int)occ,
                  (!has_shmem ? (int)plan.shared_memory_size : 0),
                  (int)ptx_version);
        }
      }
    }

    ////////////////////
    //  Variadic code
    ////////////////////

#if 0
    template<class... Args>
    static cuda_optional<int> THRUST_RUNTIME_FUNCTION
    get_max_blocks_per_sm(AgentPlan plan)
    {
      return max_blocks_per_sm_impl(_kernel_agent<Agent, Args...>, plan.block_threads);
    }
#else
    template<class _0>
    static cuda_optional<int> THRUST_RUNTIME_FUNCTION
    get_max_blocks_per_sm(AgentPlan plan)
    {
      void (*ptr)(_0) = _kernel_agent<Agent, _0>;
      return max_blocks_per_sm_impl(ptr, plan.block_threads);
    }
    template<class _0, class _1>
    static cuda_optional<int> THRUST_RUNTIME_FUNCTION
    get_max_blocks_per_sm(AgentPlan plan)
    {
      void (*ptr)(_0, _1) = _kernel_agent<Agent, _0, _1>;
      return max_blocks_per_sm_impl(ptr, plan.block_threads);
    }
    template<class _0, class _1, class _2>
    static cuda_optional<int> THRUST_RUNTIME_FUNCTION
    get_max_blocks_per_sm(AgentPlan plan)
    {
      void (*ptr)(_0,_1,_2) = _kernel_agent<Agent, _0, _1, _2>;
      return max_blocks_per_sm_impl(ptr, plan.block_threads);
    }
    template<class _0, class _1, class _2, class _3>
    static cuda_optional<int> THRUST_RUNTIME_FUNCTION
    get_max_blocks_per_sm(AgentPlan plan)
    {
      void (*ptr)(_0,_1,_2,_3) = _kernel_agent<Agent, _0, _1, _2,_3>;
      return max_blocks_per_sm_impl(ptr, plan.block_threads);
    }
    template<class _0, class _1, class _2, class _3, class _4>
    static cuda_optional<int> THRUST_RUNTIME_FUNCTION
    get_max_blocks_per_sm(AgentPlan plan)
    {
      void (*ptr)(_0,_1,_2,_3,_4) = _kernel_agent<Agent, _0, _1, _2,_3,_4>;
      return max_blocks_per_sm_impl(ptr, plan.block_threads);
    }
    template<class _0, class _1, class _2, class _3, class _4, class _5>
    static cuda_optional<int> THRUST_RUNTIME_FUNCTION
    get_max_blocks_per_sm(AgentPlan plan)
    {
      void (*ptr)(_0,_1,_2,_3,_4,_5) = _kernel_agent<Agent, _0, _1, _2,_3,_4,_5>;
      return max_blocks_per_sm_impl(ptr, plan.block_threads);
    }
    template<class _0, class _1, class _2, class _3, class _4, class _5, class _6>
    static cuda_optional<int> THRUST_RUNTIME_FUNCTION
    get_max_blocks_per_sm(AgentPlan plan)
    {
      void (*ptr)(_0,_1,_2,_3,_4,_5,_6) = _kernel_agent<Agent, _0, _1, _2,_3,_4,_5,_6>;
      return max_blocks_per_sm_impl(ptr, plan.block_threads);
    }
    template<class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7>
    static cuda_optional<int> THRUST_RUNTIME_FUNCTION
    get_max_blocks_per_sm(AgentPlan plan)
    {
      void (*ptr)(_0,_1,_2,_3,_4,_5,_6,_7) = _kernel_agent<Agent, _0, _1, _2,_3,_4,_5,_6,_7>;
      return max_blocks_per_sm_impl(ptr, plan.block_threads);
    }
    template<class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8>
    static cuda_optional<int> THRUST_RUNTIME_FUNCTION
    get_max_blocks_per_sm(AgentPlan plan)
    {
      void (*ptr)(_0,_1,_2,_3,_4,_5,_6,_7,_8) = _kernel_agent<Agent, _0, _1, _2,_3,_4,_5,_6,_7,_8>;
      return max_blocks_per_sm_impl(ptr, plan.block_threads);
    }
    template<class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9>
    static cuda_optional<int> THRUST_RUNTIME_FUNCTION
    get_max_blocks_per_sm(AgentPlan plan)
    {
      void (*ptr)(_0,_1,_2,_3,_4,_5,_6,_7,_8,_9) = _kernel_agent<Agent, _0, _1, _2,_3,_4,_5,_6,_7,_8,_9>;
      return max_blocks_per_sm_impl(ptr, plan.block_threads);
    }
    template<class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _A>
    static cuda_optional<int> THRUST_RUNTIME_FUNCTION
    get_max_blocks_per_sm(AgentPlan plan)
    {
      void (*ptr)(_0,_1,_2,_3,_4,_5,_6,_7,_8,_9,_A) = _kernel_agent<Agent, _0, _1, _2,_3,_4,_5,_6,_7,_8,_9,_A>;
      return max_blocks_per_sm_impl(ptr, plan.block_threads);
    }
    template<class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _A, class _B>
    static cuda_optional<int> THRUST_RUNTIME_FUNCTION
    get_max_blocks_per_sm(AgentPlan plan)
    {
      void (*ptr)(_0,_1,_2,_3,_4,_5,_6,_7,_8,_9,_A,_B) = _kernel_agent<Agent, _0, _1, _2,_3,_4,_5,_6,_7,_8,_9,_A,_B>;
      return max_blocks_per_sm_impl(ptr, plan.block_threads);
    }
    template<class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _A, class _B, class _C>
    static cuda_optional<int> THRUST_RUNTIME_FUNCTION
    get_max_blocks_per_sm(AgentPlan plan)
    {
      void (*ptr)(_0,_1,_2,_3,_4,_5,_6,_7,_8,_9,_A,_B,_C) = _kernel_agent<Agent, _0, _1, _2,_3,_4,_5,_6,_7,_8,_9,_A,_B,_C>;
      return max_blocks_per_sm_impl(ptr, plan.block_threads);
    }
    template<class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _A, class _B, class _C, class _D>
    static cuda_optional<int> THRUST_RUNTIME_FUNCTION
    get_max_blocks_per_sm(AgentPlan plan)
    {
      void (*ptr)(_0,_1,_2,_3,_4,_5,_6,_7,_8,_9,_A,_B,_C,_D) = _kernel_agent<Agent, _0, _1, _2,_3,_4,_5,_6,_7,_8,_9,_A,_B,_C,_D>;
      return max_blocks_per_sm_impl(ptr, plan.block_threads);
    }
    template<class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _A, class _B, class _C, class _D, class _E>
    static cuda_optional<int> THRUST_RUNTIME_FUNCTION
    get_max_blocks_per_sm(AgentPlan plan)
    {
      void (*ptr)(_0,_1,_2,_3,_4,_5,_6,_7,_8,_9,_A,_B,_C,_D,_E) = _kernel_agent<Agent, _0, _1, _2,_3,_4,_5,_6,_7,_8,_9,_A,_B,_C,_D,_E>;
      return max_blocks_per_sm_impl(ptr, plan.block_threads);
    }
#endif



#if 0

    // If we are guaranteed to have enough shared memory 
    // don't compile other kernel which accepts pointer
    // and save on compilations
    template <class... Args>
    void CUB_RUNTIME_FUNCTION
    launch_impl(detail::true_type, Args... args) const
    {
      assert(vshmem == NULL);
      print_info(_kernel_agent<Agent, Args...>);
      launcher::triple_chevron(grid, plan.block_threads, plan.shared_memory_size, stream)
          .doit(_kernel_agent<Agent, Args...>, args...);
    }
    
    // If there is a risk of not having enough shared memory 
    // we have no choice but to compile two kernels:
    // one which uses shared memory in case at runtime we find that we actually
    // to have enough
    // other which accepts global memory pointer for temporary storage
    // in case there is not enough hw shared memory 
    template <class... Args>
    void CUB_RUNTIME_FUNCTION
    launch_impl(detail::false_type, Args... args) const
    {
      if (has_shmem)
      {
        launch_impl(detail::true_type(), args...);
      }
      else
      {
        assert(vshmem != NULL);
        print_info(_kernel_agent_vshmem<Agent, Args...>);
        launcher::triple_chevron(grid, plan.block_threads, 0, stream)
            .doit(_kernel_agent_vshmem<Agent, Args...>, vshmem, args...);
      }
    }

    template <class... Args>
    void CUB_RUNTIME_FUNCTION
    launch(Args... args) const
    {
      launch_impl(has_enough_shmem_t(),args...);
      sync();
    }
#else
    template <class _0>
    void CUB_RUNTIME_FUNCTION
    launch_impl(detail::false_type, _0 x0) const
    {
      if (has_shmem)
      {
        launch_impl(detail::true_type(), x0);
      }
      else
      {
        assert(vshmem != NULL);
        void (*ptr)(char*, _0) = _kernel_agent_vshmem<Agent, _0>;
        print_info(ptr);
        launcher::triple_chevron(grid, plan.block_threads, 0, stream)
          .doit(ptr, vshmem, x0);
      }
    }
    template <class _0, class _1>
    void CUB_RUNTIME_FUNCTION
    launch_impl(detail::false_type, _0 x0, _1 x1) const
    {
      if (has_shmem)
      {
        launch_impl(detail::true_type(), x0,x1);
      }
      else
      {
        assert(vshmem != NULL);
        void (*ptr)(char*, _0,_1) = _kernel_agent_vshmem<Agent, _0,_1>;
        print_info(ptr);
        launcher::triple_chevron(grid, plan.block_threads, 0, stream)
          .doit(ptr, vshmem, x0, x1);
      }
    }
    template <class _0, class _1, class _2>
    void CUB_RUNTIME_FUNCTION
    launch_impl(detail::false_type, _0 x0, _1 x1, _2 x2) const
    {
      if (has_shmem)
      {
        launch_impl(detail::true_type(), x0,x1,x2);
      }
      else
      {
        assert(vshmem != NULL);
        void (*ptr)(char*, _0,_1,_2) = _kernel_agent_vshmem<Agent, _0,_1,_2>;
        print_info(ptr);
        launcher::triple_chevron(grid, plan.block_threads, 0, stream)
          .doit(ptr, vshmem, x0, x1, x2);
      }
    }
    template <class _0, class _1, class _2, class _3>
    void CUB_RUNTIME_FUNCTION
    launch_impl(detail::false_type, _0 x0, _1 x1, _2 x2, _3 x3) const
    {
      if (has_shmem)
      {
        launch_impl(detail::true_type(), x0,x1,x2,x3);
      }
      else
      {
        assert(vshmem != NULL);
        void (*ptr)(char*, _0,_1,_2,_3) = _kernel_agent_vshmem<Agent, _0,_1,_2,_3>;
        print_info(ptr);
        launcher::triple_chevron(grid, plan.block_threads, 0, stream)
          .doit(ptr, vshmem, x0, x1, x2, x3);
      }
    }
    template <class _0, class _1, class _2, class _3, class _4>
    void CUB_RUNTIME_FUNCTION
    launch_impl(detail::false_type, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4) const
    {
      if (has_shmem)
      {
        launch_impl(detail::true_type(), x0,x1,x2,x3,x4);
      }
      else
      {
        assert(vshmem != NULL);
        void (*ptr)(char*, _0,_1,_2,_3,_4) = _kernel_agent_vshmem<Agent, _0,_1,_2,_3,_4>;
        print_info(ptr);
        launcher::triple_chevron(grid, plan.block_threads, 0, stream)
          .doit(ptr, vshmem, x0, x1, x2, x3, x4);
      }
    }
    template <class _0, class _1, class _2, class _3, class _4, class _5>
    void CUB_RUNTIME_FUNCTION
    launch_impl(detail::false_type, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5) const
    {
      if (has_shmem)
      {
        launch_impl(detail::true_type(), x0,x1,x2,x3,x4,x5);
      }
      else
      {
        assert(vshmem != NULL);
        void (*ptr)(char*, _0,_1,_2,_3,_4,_5) = _kernel_agent_vshmem<Agent, _0,_1,_2,_3,_4,_5>;
        print_info(ptr);
        launcher::triple_chevron(grid, plan.block_threads, 0, stream)
          .doit(ptr, vshmem, x0, x1, x2, x3, x4, x5);
      }
    }
    template <class _0, class _1, class _2, class _3, class _4, class _5, class _6>
    void CUB_RUNTIME_FUNCTION
    launch_impl(detail::false_type, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6) const
    {
      if (has_shmem)
      {
        launch_impl(detail::true_type(), x0,x1,x2,x3,x4,x5,x6);
      }
      else
      {
        assert(vshmem != NULL);
        void (*ptr)(char*, _0,_1,_2,_3,_4,_5,_6) = _kernel_agent_vshmem<Agent, _0,_1,_2,_3,_4,_5,_6>;
        print_info(ptr);
        launcher::triple_chevron(grid, plan.block_threads, 0, stream)
          .doit(ptr, vshmem, x0, x1, x2, x3, x4, x5, x6);
      }
    }
    template <class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7>
    void CUB_RUNTIME_FUNCTION
    launch_impl(detail::false_type, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7) const
    {
      if (has_shmem)
      {
        launch_impl(detail::true_type(), x0,x1,x2,x3,x4,x5,x6,x7);
      }
      else
      {
        assert(vshmem != NULL);
        void (*ptr)(char*, _0,_1,_2,_3,_4,_5,_6,_7) = _kernel_agent_vshmem<Agent, _0,_1,_2,_3,_4,_5,_6,_7>;
        print_info(ptr);
        launcher::triple_chevron(grid, plan.block_threads, 0, stream)
          .doit(ptr, vshmem, x0, x1, x2, x3, x4, x5, x6, x7);
      }
    }
    template <class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8>
    void CUB_RUNTIME_FUNCTION
    launch_impl(detail::false_type, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8) const
    {
      if (has_shmem)
      {
        launch_impl(detail::true_type(), x0,x1,x2,x3,x4,x5,x6,x7,x8);
      }
      else
      {
        assert(vshmem != NULL);
        void (*ptr)(char*, _0,_1,_2,_3,_4,_5,_6,_7,_8) = _kernel_agent_vshmem<Agent, _0,_1,_2,_3,_4,_5,_6,_7,_8>;
        launcher::triple_chevron(grid, plan.block_threads, 0, stream)
          .doit(ptr, vshmem, x0, x1, x2, x3, x4, x5, x6, x7, x8);
      }
    }
    template <class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9>
    void CUB_RUNTIME_FUNCTION
    launch_impl(detail::false_type, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8, _9 x9) const
    {
      if (has_shmem)
      {
        launch_impl(detail::true_type(), x0,x1,x2,x3,x4,x5,x6,x7,x8,x9);
      }
      else
      {
        assert(vshmem != NULL);
        void (*ptr)(char*, _0,_1,_2,_3,_4,_5,_6,_7,_8,_9) = _kernel_agent_vshmem<Agent, _0,_1,_2,_3,_4,_5,_6,_7,_8,_9>;
        print_info(ptr);
        launcher::triple_chevron(grid, plan.block_threads, 0, stream)
          .doit(ptr, vshmem, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9);
      }
    }
    template <class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _A>
    void CUB_RUNTIME_FUNCTION
    launch_impl(detail::false_type, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8, _9 x9,_A xA) const
    {
      if (has_shmem)
      {
        launch_impl(detail::true_type(), x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,xA);
      }
      else
      {
        assert(vshmem != NULL);
        void (*ptr)(char*, _0,_1,_2,_3,_4,_5,_6,_7,_8,_9,_A) = _kernel_agent_vshmem<Agent, _0,_1,_2,_3,_4,_5,_6,_7,_8,_9,_A>;
        print_info(ptr);
        launcher::triple_chevron(grid, plan.block_threads, 0, stream)
          .doit(ptr, vshmem, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, xA);
      }
    }
    template <class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _A, class _B>
    void CUB_RUNTIME_FUNCTION
    launch_impl(detail::false_type, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8, _9 x9,_A xA,_B xB) const
    {
      if (has_shmem)
      {
        launch_impl(detail::true_type(), x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,xA,xB);
      }
      else
      {
        assert(vshmem != NULL);
        void (*ptr)(char*, _0,_1,_2,_3,_4,_5,_6,_7,_8,_9,_A,_B) = _kernel_agent_vshmem<Agent, _0,_1,_2,_3,_4,_5,_6,_7,_8,_9,_A,_B>;
        print_info(ptr);
        launcher::triple_chevron(grid, plan.block_threads, 0, stream)
          .doit(ptr, vshmem, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, xA, xB);
      }
    }
    template <class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _A, class _B, class _C>
    void CUB_RUNTIME_FUNCTION
    launch_impl(detail::false_type, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8, _9 x9,_A xA,_B xB,_C xC) const
    {
      if (has_shmem)
      {
        launch_impl(detail::true_type(), x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,xA,xB,xC);
      }
      else
      {
        assert(vshmem != NULL);
        void (*ptr)(char*, _0,_1,_2,_3,_4,_5,_6,_7,_8,_9,_A,_B,_C) = _kernel_agent_vshmem<Agent, _0,_1,_2,_3,_4,_5,_6,_7,_8,_9,_A,_B,_C>;
        print_info(ptr);
        launcher::triple_chevron(grid, plan.block_threads, 0, stream)
          .doit(ptr, vshmem, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, xA, xB, xC);
      }
    }
    template <class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _A, class _B, class _C, class _D>
    void CUB_RUNTIME_FUNCTION
    launch_impl(detail::false_type, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8, _9 x9,_A xA,_B xB,_C xC,_D xD) const
    {
      if (has_shmem)
      {
        launch_impl(detail::true_type(), x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,xA,xB,xC,xD);
      }
      else
      {
        assert(vshmem != NULL);
        void (*ptr)(char*, _0,_1,_2,_3,_4,_5,_6,_7,_8,_9,_A,_B,_C,_D) = _kernel_agent_vshmem<Agent, _0,_1,_2,_3,_4,_5,_6,_7,_8,_9,_A,_B,_C,_D>;
        print_info(ptr);
        launcher::triple_chevron(grid, plan.block_threads, 0, stream)
          .doit(ptr, vshmem, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, xA, xB, xC, xD);
      }
    }
    template <class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _A, class _B, class _C, class _D, class _E>
    void CUB_RUNTIME_FUNCTION
    launch_impl(detail::false_type, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8, _9 x9,_A xA,_B xB,_C xC,_D xD,_E xE) const
    {
      if (has_shmem)
      {
        launch_impl(detail::true_type(), x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,xA,xB,xC,xD,xE);
      }
      else
      {
        assert(vshmem != NULL);
        void (*ptr)(char*, _0,_1,_2,_3,_4,_5,_6,_7,_8,_9,_A,_B,_C,_D,_E) = _kernel_agent_vshmem<Agent, _0,_1,_2,_3,_4,_5,_6,_7,_8,_9,_A,_B,_C,_D,_E>;
        print_info(ptr);
        launcher::triple_chevron(grid, plan.block_threads, 0, stream)
          .doit(ptr, vshmem, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, xA, xB, xC, xD,xE);
      }
    }

    ////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////

    template <class _0>
    void CUB_RUNTIME_FUNCTION
    launch_impl(detail::true_type, _0 x0) const
    {
      assert(vshmem == NULL);
      void (*ptr)(_0) = _kernel_agent<Agent, _0>;
      print_info(ptr);
      launcher::triple_chevron(grid, plan.block_threads, plan.shared_memory_size, stream)
          .doit(ptr);
    }
    template <class _0, class _1>
    void CUB_RUNTIME_FUNCTION
    launch_impl(detail::true_type, _0 x0, _1 x1) const
    {
      assert(vshmem == NULL);
      void (*ptr)(_0, _1) = _kernel_agent<Agent, _0, _1>;
      print_info(ptr);
      launcher::triple_chevron(grid, plan.block_threads, plan.shared_memory_size, stream)
          .doit(ptr, x0, x1);
    }
    template <class _0, class _1, class _2>
    void CUB_RUNTIME_FUNCTION
    launch_impl(detail::true_type, _0 x0, _1 x1, _2 x2) const
    {
      assert(vshmem == NULL);
      void (*ptr)(_0,_1,_2) = _kernel_agent<Agent, _0, _1, _2>;
      print_info(ptr);
      launcher::triple_chevron(grid, plan.block_threads, plan.shared_memory_size, stream)
          .doit(ptr, x0, x1, x2);
    }
    template <class _0, class _1, class _2, class _3>
    void CUB_RUNTIME_FUNCTION
    launch_impl(detail::true_type, _0 x0, _1 x1, _2 x2, _3 x3) const
    {
      assert(vshmem == NULL);
      void (*ptr)(_0,_1,_2,_3) = _kernel_agent<Agent, _0, _1, _2,_3>;
      print_info(ptr);
      launcher::triple_chevron(grid, plan.block_threads, plan.shared_memory_size, stream)
          .doit(ptr, x0, x1, x2, x3);
    }
    template <class _0, class _1, class _2, class _3, class _4>
    void CUB_RUNTIME_FUNCTION
    launch_impl(detail::true_type, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4) const
    {
      assert(vshmem == NULL);
      void (*ptr)(_0,_1,_2,_3,_4) = _kernel_agent<Agent, _0, _1, _2,_3,_4>;
      print_info(ptr);
      launcher::triple_chevron(grid, plan.block_threads, plan.shared_memory_size, stream)
          .doit(ptr, x0, x1, x2, x3, x4);
    }
    template <class _0, class _1, class _2, class _3, class _4, class _5>
    void CUB_RUNTIME_FUNCTION
    launch_impl(detail::true_type, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5) const
    {
      assert(vshmem == NULL);
      void (*ptr)(_0,_1,_2,_3,_4,_5) = _kernel_agent<Agent, _0, _1, _2,_3,_4,_5>;
      print_info(ptr);
      launcher::triple_chevron(grid, plan.block_threads, plan.shared_memory_size, stream)
          .doit(ptr, x0, x1, x2, x3, x4, x5);
    }
    template <class _0, class _1, class _2, class _3, class _4, class _5, class _6>
    void CUB_RUNTIME_FUNCTION
    launch_impl(detail::true_type, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6) const
    {
      assert(vshmem == NULL);
      void (*ptr)(_0,_1,_2,_3,_4,_5,_6) = _kernel_agent<Agent, _0, _1, _2,_3,_4,_5,_6>;
      print_info(ptr);
      launcher::triple_chevron(grid, plan.block_threads, plan.shared_memory_size, stream)
          .doit(ptr, x0, x1, x2, x3, x4, x5, x6);
    }
    template <class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7>
    void CUB_RUNTIME_FUNCTION
    launch_impl(detail::true_type, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7) const
    {
      assert(vshmem == NULL);
      void (*ptr)(_0,_1,_2,_3,_4,_5,_6,_7) = _kernel_agent<Agent, _0, _1, _2,_3,_4,_5,_6,_7>;
      print_info(ptr);
      launcher::triple_chevron(grid, plan.block_threads, plan.shared_memory_size, stream)
          .doit(ptr, x0, x1, x2, x3, x4, x5, x6, x7);
    }
    template <class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8>
    void CUB_RUNTIME_FUNCTION
    launch_impl(detail::true_type, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8) const
    {
      assert(vshmem == NULL);
      void (*ptr)(_0,_1,_2,_3,_4,_5,_6,_7,_8) = _kernel_agent<Agent, _0, _1, _2,_3,_4,_5,_6,_7,_8>;
      print_info(ptr);
      launcher::triple_chevron(grid, plan.block_threads, plan.shared_memory_size, stream)
          .doit(ptr, x0, x1, x2, x3, x4, x5, x6, x7, x8);
    }
    template <class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9>
    void CUB_RUNTIME_FUNCTION
    launch_impl(detail::true_type, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8, _9 x9) const
    {
      assert(vshmem == NULL);
      void (*ptr)(_0,_1,_2,_3,_4,_5,_6,_7,_8,_9) = _kernel_agent<Agent, _0, _1, _2,_3,_4,_5,_6,_7,_8,_9>;
      print_info(ptr);
      launcher::triple_chevron(grid, plan.block_threads, plan.shared_memory_size, stream)
          .doit(ptr, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9);
    }
    template <class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _A>
    void CUB_RUNTIME_FUNCTION
    launch_impl(detail::true_type, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8, _9 x9, _A xA) const
    {
      assert(vshmem == NULL);
      void (*ptr)(_0,_1,_2,_3,_4,_5,_6,_7,_8,_9,_A) = _kernel_agent<Agent, _0, _1, _2,_3,_4,_5,_6,_7,_8,_9,_A>;
      print_info(ptr);
      launcher::triple_chevron(grid, plan.block_threads, plan.shared_memory_size, stream)
          .doit(ptr, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, xA);
    }
    template <class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _A, class _B>
    void CUB_RUNTIME_FUNCTION
    launch_impl(detail::true_type, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8, _9 x9, _A xA, _B xB) const
    {
      assert(vshmem == NULL);
      void (*ptr)(_0,_1,_2,_3,_4,_5,_6,_7,_8,_9,_A,_B) = _kernel_agent<Agent, _0, _1, _2,_3,_4,_5,_6,_7,_8,_9,_A,_B>;
      print_info(ptr);
      launcher::triple_chevron(grid, plan.block_threads, plan.shared_memory_size, stream)
          .doit(ptr, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, xA, xB);
    }
    template <class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _A, class _B, class _C>
    void CUB_RUNTIME_FUNCTION
    launch_impl(detail::true_type, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8, _9 x9, _A xA, _B xB, _C xC) const
    {
      assert(vshmem == NULL);
      void (*ptr)(_0,_1,_2,_3,_4,_5,_6,_7,_8,_9,_A,_B,_C) = _kernel_agent<Agent, _0, _1, _2,_3,_4,_5,_6,_7,_8,_9,_A,_B,_C>;
      print_info(ptr);
      launcher::triple_chevron(grid, plan.block_threads, plan.shared_memory_size, stream)
          .doit(ptr, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, xA, xB, xC);
    }
    template <class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _A, class _B, class _C, class _D>
    void CUB_RUNTIME_FUNCTION
    launch_impl(detail::true_type, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8, _9 x9, _A xA, _B xB, _C xC, _D xD) const
    {
      assert(vshmem == NULL);
      void (*ptr)(_0,_1,_2,_3,_4,_5,_6,_7,_8,_9,_A,_B,_C,_D) = _kernel_agent<Agent, _0, _1, _2,_3,_4,_5,_6,_7,_8,_9,_A,_B,_C,_D>;
      print_info(ptr);
      launcher::triple_chevron(grid, plan.block_threads, plan.shared_memory_size, stream)
          .doit(ptr, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, xA, xB, xC, xD);
    }
    template <class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _A, class _B, class _C, class _D, class _E>
    void CUB_RUNTIME_FUNCTION
    launch_impl(detail::true_type, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8, _9 x9, _A xA, _B xB, _C xC, _D xD, _E xE) const
    {
      assert(vshmem == NULL);
      void (*ptr)(_0,_1,_2,_3,_4,_5,_6,_7,_8,_9,_A,_B,_C,_D,_E) = _kernel_agent<Agent, _0, _1, _2,_3,_4,_5,_6,_7,_8,_9,_A,_B,_C,_D,_E>;
      print_info(ptr);
      launcher::triple_chevron(grid, plan.block_threads, plan.shared_memory_size, stream)
          .doit(ptr,x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, xA, xB, xC, xD, xE);
    }

    ////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////
    
    template <class _0>
    void CUB_RUNTIME_FUNCTION
    launch(_0 x0) const
    {
      launch_impl(has_enough_shmem_t(), x0);
      sync();
    }
    template <class _0, class _1>
    void CUB_RUNTIME_FUNCTION
    launch(_0 x0, _1 x1) const
    {
      launch_impl(has_enough_shmem_t(), x0, x1);
      sync();
    }
    template <class _0, class _1, class _2>
    void CUB_RUNTIME_FUNCTION
    launch(_0 x0, _1 x1, _2 x2) const
    {
      launch_impl(has_enough_shmem_t(), x0, x1, x2);
      sync();
    }
    template <class _0, class _1, class _2, class _3>
    void CUB_RUNTIME_FUNCTION
    launch(_0 x0, _1 x1, _2 x2, _3 x3) const
    {
      launch_impl(has_enough_shmem_t(), x0, x1, x2, x3);
      sync();
    }
    template <class _0, class _1, class _2, class _3, class _4>
    void CUB_RUNTIME_FUNCTION
    launch(_0 x0, _1 x1, _2 x2, _3 x3, _4 x4) const
    {
      launch_impl(has_enough_shmem_t(), x0, x1, x2, x3, x4);
      sync();
    }
    template <class _0, class _1, class _2, class _3, class _4, class _5>
    void CUB_RUNTIME_FUNCTION
    launch(_0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5) const
    {
      launch_impl(has_enough_shmem_t(), x0, x1, x2, x3, x4, x5);
      sync();
    }
    template <class _0, class _1, class _2, class _3, class _4, class _5, class _6>
    void CUB_RUNTIME_FUNCTION
    launch(_0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6) const
    {
      launch_impl(has_enough_shmem_t(), x0, x1, x2, x3, x4, x5, x6);
      sync();
    }
    template <class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7>
    void CUB_RUNTIME_FUNCTION
    launch(_0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7) const
    {
      launch_impl(has_enough_shmem_t(), x0, x1, x2, x3, x4, x5, x6, x7);
      sync();
    }
    template <class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8>
    void CUB_RUNTIME_FUNCTION
    launch(_0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8) const
    {
      launch_impl(has_enough_shmem_t(), x0, x1, x2, x3, x4, x5, x6, x7, x8);
      sync();
    }
    template <class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9>
    void CUB_RUNTIME_FUNCTION
    launch(_0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8, _9 x9) const
    {
      launch_impl(has_enough_shmem_t(), x0, x1, x2, x3, x4, x5, x6, x7, x8, x9);
      sync();
    }
    template <class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _A>
    void CUB_RUNTIME_FUNCTION
    launch(_0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8, _9 x9, _A xA) const
    {
      launch_impl(has_enough_shmem_t(), x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, xA);
      sync();
    }
    template <class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _A, class _B>
    void CUB_RUNTIME_FUNCTION
    launch(_0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8, _9 x9, _A xA, _B xB) const
    {
      launch_impl(has_enough_shmem_t(), x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, xA, xB);
      sync();
    }
    template <class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _A, class _B, class _C>
    void CUB_RUNTIME_FUNCTION
    launch(_0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8, _9 x9, _A xA, _B xB, _C xC) const
    {
      launch_impl(has_enough_shmem_t(), x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, xA, xB, xC);
      sync();
    }
    template <class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _A, class _B, class _C, class _D>
    void CUB_RUNTIME_FUNCTION
    launch(_0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8, _9 x9, _A xA, _B xB, _C xC, _D xD) const
    {
      launch_impl(has_enough_shmem_t(), x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, xA, xB, xC, xD);
      sync();
    }
    template <class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _A, class _B, class _C, class _D, class _E>
    void CUB_RUNTIME_FUNCTION
    launch(_0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8, _9 x9, _A xA, _B xB, _C xC, _D xD, _E xE) const
    {
      launch_impl(has_enough_shmem_t(), x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, xA, xB, xC, xD, xE);
      sync();
    }
#endif


  };

}    // namespace core
}
END_NS_THRUST
#endif
