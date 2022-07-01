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

#include <thrust/detail/config.h>

#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC
#include <thrust/system/cuda/detail/guarded_cuda_runtime_api.h>
#include <thrust/system/cuda/detail/core/triple_chevron_launch.h>
#include <thrust/system/cuda/detail/core/util.h>

#include <cub/detail/device_synchronize.cuh>
#include <cub/detail/ptx_dispatch.cuh>

#include <nv/target>

#include <cassert>

THRUST_NAMESPACE_BEGIN
namespace cuda_cub {
namespace core {

#ifndef _NVHPC_CUDA

  template <class Agent, class PtxPlan, class... Args>
   __global__ __launch_bounds__(PtxPlan::BLOCK_THREADS)
  void _kernel_agent(Args... args)
  {
    extern __shared__ char shmem[];
    Agent::template entry<PtxPlan>(args..., shmem);
  }

  template <class Agent, class PtxPlan, class... Args>
  __global__ __launch_bounds__(PtxPlan::BLOCK_THREADS)
  void _kernel_agent_vshmem(char* vshmem, Args... args)
  {
    extern __shared__ char shmem[];
    vshmem = vshmem == NULL ? shmem : vshmem + blockIdx.x * temp_storage_size<PtxPlan>::value;
    Agent::template entry<PtxPlan>(args..., vshmem);
  }

#else

  // WAR nvc++ bug (NVBug 3714031) by explicitly expanding the variadic
  // templates:

  // clang-format off
  template <class Agent, class PtxPlan, class _0>
  void __global__ __launch_bounds__(PtxPlan::BLOCK_THREADS)
    _kernel_agent(_0 x0)
  {
    extern __shared__ char shmem[];
    Agent::template entry<PtxPlan>(x0, shmem);
  }
  template <class Agent, class PtxPlan, class _0, class _1>
  void __global__ __launch_bounds__(PtxPlan::BLOCK_THREADS)
    _kernel_agent(_0 x0, _1 x1)
  {
    extern __shared__ char shmem[];
    Agent::template entry<PtxPlan>(x0, x1, shmem);
  }
  template <class Agent, class PtxPlan, class _0, class _1, class _2>
  void __global__ __launch_bounds__(PtxPlan::BLOCK_THREADS)
    _kernel_agent(_0 x0, _1 x1, _2 x2)
  {
    extern __shared__ char shmem[];
    Agent::template entry<PtxPlan>(x0, x1, x2, shmem);
  }
  template <class Agent, class PtxPlan, class _0, class _1, class _2, class _3>
  void __global__ __launch_bounds__(PtxPlan::BLOCK_THREADS)
    _kernel_agent(_0 x0, _1 x1, _2 x2, _3 x3)
  {
    extern __shared__ char shmem[];
    Agent::template entry<PtxPlan>(x0, x1, x2, x3, shmem);
  }
  template <class Agent, class PtxPlan, class _0, class _1, class _2, class _3, class _4>
  void __global__ __launch_bounds__(PtxPlan::BLOCK_THREADS)
    _kernel_agent(_0 x0, _1 x1, _2 x2, _3 x3, _4 x4)
  {
    extern __shared__ char shmem[];
    Agent::template entry<PtxPlan>(x0, x1, x2, x3, x4, shmem);
  }
  template <class Agent, class PtxPlan, class _0, class _1, class _2, class _3, class _4, class _5>
  void __global__ __launch_bounds__(PtxPlan::BLOCK_THREADS)
    _kernel_agent(_0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5)
  {
    extern __shared__ char shmem[];
    Agent::template entry<PtxPlan>(x0, x1, x2, x3, x4, x5, shmem);
  }
  template <class Agent, class PtxPlan, class _0, class _1, class _2, class _3, class _4, class _5, class _6>
  void __global__ __launch_bounds__(PtxPlan::BLOCK_THREADS)
    _kernel_agent(_0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6)
  {
    extern __shared__ char shmem[];
    Agent::template entry<PtxPlan>(x0, x1, x2, x3, x4, x5, x6, shmem);
  }
  template <class Agent, class PtxPlan, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7>
  void __global__ __launch_bounds__(PtxPlan::BLOCK_THREADS)
    _kernel_agent(_0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7)
  {
    extern __shared__ char shmem[];
    Agent::template entry<PtxPlan>(x0, x1, x2, x3, x4, x5, x6, x7, shmem);
  }
  template <class Agent, class PtxPlan, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8>
  void __global__ __launch_bounds__(PtxPlan::BLOCK_THREADS)
    _kernel_agent(_0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8)
  {
    extern __shared__ char shmem[];
    Agent::template entry<PtxPlan>(x0, x1, x2, x3, x4, x5, x6, x7, x8, shmem);
  }
  template <class Agent, class PtxPlan, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9>
  void __global__ __launch_bounds__(PtxPlan::BLOCK_THREADS)
    _kernel_agent(_0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8, _9 x9)
  {
    extern __shared__ char shmem[];
    Agent::template entry<PtxPlan>(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, shmem);
  }
  template <class Agent, class PtxPlan, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _xA>
  void __global__ __launch_bounds__(PtxPlan::BLOCK_THREADS)
    _kernel_agent(_0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8, _9 x9, _xA xA)
  {
    extern __shared__ char shmem[];
    Agent::template entry<PtxPlan>(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, xA, shmem);
  }
  template <class Agent, class PtxPlan, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _xA, class _xB>
  void __global__ __launch_bounds__(PtxPlan::BLOCK_THREADS)
    _kernel_agent(_0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8, _9 x9, _xA xA, _xB xB)
  {
    extern __shared__ char shmem[];
    Agent::template entry<PtxPlan>(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, xA, xB, shmem);
  }
  template <class Agent, class PtxPlan, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _xA, class _xB, class _xC>
  void __global__ __launch_bounds__(PtxPlan::BLOCK_THREADS)
    _kernel_agent(_0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8, _9 x9, _xA xA, _xB xB, _xC xC)
  {
    extern __shared__ char shmem[];
    Agent::template entry<PtxPlan>(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, xA, xB, xC, shmem);
  }
  template <class Agent, class PtxPlan, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _xA, class _xB, class _xC, class _xD>
  void __global__ __launch_bounds__(PtxPlan::BLOCK_THREADS)
    _kernel_agent(_0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8, _9 x9, _xA xA, _xB xB, _xC xC, _xD xD)
  {
    extern __shared__ char shmem[];
    Agent::template entry<PtxPlan>(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, xA, xB, xC, xD, shmem);
  }
  template <class Agent, class PtxPlan, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _xA, class _xB, class _xC, class _xD, class _xE>
  void __global__ __launch_bounds__(PtxPlan::BLOCK_THREADS)
    _kernel_agent(_0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8, _9 x9, _xA xA, _xB xB, _xC xC, _xD xD, _xE xE)
  {
    extern __shared__ char shmem[];
    Agent::template entry<PtxPlan>(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, xA, xB, xC, xD, xE, shmem);
  }

  template <class Agent, class PtxPlan, class _0>
  void __global__ __launch_bounds__(PtxPlan::BLOCK_THREADS)
    _kernel_agent_vshmem(char* vshmem, _0 x0)
  {
    extern __shared__ char shmem[];
    vshmem = vshmem == NULL ? shmem : vshmem + blockIdx.x * temp_storage_size<PtxPlan>::value;
    Agent::template entry<PtxPlan>(x0, vshmem);
  }
  template <class Agent, class PtxPlan, class _0, class _1>
  void __global__ __launch_bounds__(PtxPlan::BLOCK_THREADS)
    _kernel_agent_vshmem(char* vshmem, _0 x0, _1 x1)
  {
    extern __shared__ char shmem[];
    vshmem = vshmem == NULL ? shmem : vshmem + blockIdx.x * temp_storage_size<PtxPlan>::value;
    Agent::template entry<PtxPlan>(x0, x1, vshmem);
  }
  template <class Agent, class PtxPlan, class _0, class _1, class _2>
  void __global__ __launch_bounds__(PtxPlan::BLOCK_THREADS)
    _kernel_agent_vshmem(char* vshmem, _0 x0, _1 x1, _2 x2)
  {
    extern __shared__ char shmem[];
    vshmem = vshmem == NULL ? shmem : vshmem + blockIdx.x * temp_storage_size<PtxPlan>::value;
    Agent::template entry<PtxPlan>(x0, x1, x2, vshmem);
  }
  template <class Agent, class PtxPlan, class _0, class _1, class _2, class _3>
  void __global__ __launch_bounds__(PtxPlan::BLOCK_THREADS)
    _kernel_agent_vshmem(char* vshmem, _0 x0, _1 x1, _2 x2, _3 x3)
  {
    extern __shared__ char shmem[];
    vshmem = vshmem == NULL ? shmem : vshmem + blockIdx.x * temp_storage_size<PtxPlan>::value;
    Agent::template entry<PtxPlan>(x0, x1, x2, x3, vshmem);
  }
  template <class Agent, class PtxPlan, class _0, class _1, class _2, class _3, class _4>
  void __global__ __launch_bounds__(PtxPlan::BLOCK_THREADS)
    _kernel_agent_vshmem(char* vshmem, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4)
  {
    extern __shared__ char shmem[];
    vshmem = vshmem == NULL ? shmem : vshmem + blockIdx.x * temp_storage_size<PtxPlan>::value;
    Agent::template entry<PtxPlan>(x0, x1, x2, x3, x4, vshmem);
  }
  template <class Agent, class PtxPlan, class _0, class _1, class _2, class _3, class _4, class _5>
  void __global__ __launch_bounds__(PtxPlan::BLOCK_THREADS)
    _kernel_agent_vshmem(char* vshmem, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5)
  {
    extern __shared__ char shmem[];
    vshmem = vshmem == NULL ? shmem : vshmem + blockIdx.x * temp_storage_size<PtxPlan>::value;
    Agent::template entry<PtxPlan>(x0, x1, x2, x3, x4, x5, vshmem);
  }
  template <class Agent, class PtxPlan, class _0, class _1, class _2, class _3, class _4, class _5, class _6>
  void __global__ __launch_bounds__(PtxPlan::BLOCK_THREADS)
    _kernel_agent_vshmem(char* vshmem, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6)
  {
    extern __shared__ char shmem[];
    vshmem = vshmem == NULL ? shmem : vshmem + blockIdx.x * temp_storage_size<PtxPlan>::value;
    Agent::template entry<PtxPlan>(x0, x1, x2, x3, x4, x5, x6, vshmem);
  }
  template <class Agent, class PtxPlan, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7>
  void __global__ __launch_bounds__(PtxPlan::BLOCK_THREADS)
    _kernel_agent_vshmem(char* vshmem, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7)
  {
    extern __shared__ char shmem[];
    vshmem = vshmem == NULL ? shmem : vshmem + blockIdx.x * temp_storage_size<PtxPlan>::value;
    Agent::template entry<PtxPlan>(x0, x1, x2, x3, x4, x5, x6, x7, vshmem);
  }
  template <class Agent, class PtxPlan, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8>
  void __global__ __launch_bounds__(PtxPlan::BLOCK_THREADS)
    _kernel_agent_vshmem(char* vshmem, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8)
  {
    extern __shared__ char shmem[];
    vshmem = vshmem == NULL ? shmem : vshmem + blockIdx.x * temp_storage_size<PtxPlan>::value;
    Agent::template entry<PtxPlan>(x0, x1, x2, x3, x4, x5, x6, x7, x8, vshmem);
  }
  template <class Agent, class PtxPlan, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9>
  void __global__ __launch_bounds__(PtxPlan::BLOCK_THREADS)
    _kernel_agent_vshmem(char* vshmem, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8, _9 x9)
  {
    extern __shared__ char shmem[];
    vshmem = vshmem == NULL ? shmem : vshmem + blockIdx.x * temp_storage_size<PtxPlan>::value;
    Agent::template entry<PtxPlan>(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, vshmem);
  }
  template <class Agent, class PtxPlan, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _xA>
  void __global__ __launch_bounds__(PtxPlan::BLOCK_THREADS)
    _kernel_agent_vshmem(char* vshmem, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8, _9 x9, _xA xA)
  {
    extern __shared__ char shmem[];
    vshmem = vshmem == NULL ? shmem : vshmem + blockIdx.x * temp_storage_size<PtxPlan>::value;
    Agent::template entry<PtxPlan>(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, xA, vshmem);
  }
  template <class Agent, class PtxPlan, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _xA, class _xB>
  void __global__ __launch_bounds__(PtxPlan::BLOCK_THREADS)
    _kernel_agent_vshmem(char* vshmem, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8, _9 x9, _xA xA, _xB xB)
  {
    extern __shared__ char shmem[];
    vshmem = vshmem == NULL ? shmem : vshmem + blockIdx.x * temp_storage_size<PtxPlan>::value;
    Agent::template entry<PtxPlan>(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, xA, xB, vshmem);
  }
  template <class Agent, class PtxPlan, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _xA, class _xB, class _xC>
  void __global__ __launch_bounds__(PtxPlan::BLOCK_THREADS)
    _kernel_agent_vshmem(char* vshmem, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8, _9 x9, _xA xA, _xB xB, _xC xC)
  {
    extern __shared__ char shmem[];
    vshmem = vshmem == NULL ? shmem : vshmem + blockIdx.x * temp_storage_size<PtxPlan>::value;
    Agent::template entry<PtxPlan>(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, xA, xB, xC, vshmem);
  }
  template <class Agent, class PtxPlan, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _xA, class _xB, class _xC, class _xD>
  void __global__ __launch_bounds__(PtxPlan::BLOCK_THREADS)
    _kernel_agent_vshmem(char* vshmem, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8, _9 x9, _xA xA, _xB xB, _xC xC, _xD xD)
  {
    extern __shared__ char shmem[];
    vshmem = vshmem == NULL ? shmem : vshmem + blockIdx.x * temp_storage_size<PtxPlan>::value;
    Agent::template entry<PtxPlan>(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, xA, xB, xC, xD, vshmem);
  }
  template <class Agent, class PtxPlan, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _xA, class _xB, class _xC, class _xD, class _xE>
  void __global__ __launch_bounds__(PtxPlan::BLOCK_THREADS)
    _kernel_agent_vshmem(char* vshmem, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8, _9 x9, _xA xA, _xB xB, _xC xC, _xD xD, _xE xE)
  {
    extern __shared__ char shmem[];
    vshmem = vshmem == NULL ? shmem : vshmem + blockIdx.x * temp_storage_size<PtxPlan>::value;
    Agent::template entry<PtxPlan>(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, xA, xB, xC, xD, xE, vshmem);
  }
  // clang-format on

#endif

  template<class Agent>
  struct AgentLauncher : Agent
  {
    core::AgentPlan plan;
    std::size_t     count;
    cudaStream_t    stream;
    char const*     name;
    unsigned int    grid;
    char*           vshmem;
    bool            has_shmem;
    std::size_t     shmem_size;

    static constexpr std::size_t MAX_SHMEM_PER_BLOCK = 48 * 1024;

    template <class Size>
    THRUST_RUNTIME_FUNCTION
    AgentLauncher(AgentPlan    plan_,
                  Size         count_,
                  cudaStream_t stream_,
                  char const*  name_)
        : plan(plan_),
          count((size_t)count_),
          stream(stream_),
          name(name_),
          grid(static_cast<unsigned int>((count + plan.items_per_tile - 1) / plan.items_per_tile)),
          vshmem(NULL),
          has_shmem((size_t)core::get_max_shared_memory_per_block() >= (size_t)plan.shared_memory_size),
          shmem_size(has_shmem ? plan.shared_memory_size : 0)
    {
      assert(count > 0);
    }

    template <class Size>
    THRUST_RUNTIME_FUNCTION
    AgentLauncher(AgentPlan    plan_,
                  Size         count_,
                  cudaStream_t stream_,
                  char*        vshmem,
                  char const*  name_)
        : plan(plan_),
          count((size_t)count_),
          stream(stream_),
          name(name_),
          grid(static_cast<unsigned int>((count + plan.items_per_tile - 1) / plan.items_per_tile)),
          vshmem(vshmem),
          has_shmem((size_t)core::get_max_shared_memory_per_block() >= (size_t)plan.shared_memory_size),
          shmem_size(has_shmem ? plan.shared_memory_size : 0)
    {
      assert(count > 0);
    }

    THRUST_RUNTIME_FUNCTION
    AgentLauncher(AgentPlan    plan_,
                  cudaStream_t stream_,
                  char const*  name_)
        : plan(plan_),
          count(0),
          stream(stream_),
          name(name_),
          grid(plan.grid_size),
          vshmem(NULL),
          has_shmem((size_t)core::get_max_shared_memory_per_block() >= (size_t)plan.shared_memory_size),
          shmem_size(has_shmem ? plan.shared_memory_size : 0)
    {
      assert(plan.grid_size > 0);
    }

    THRUST_RUNTIME_FUNCTION
    AgentLauncher(AgentPlan    plan_,
                  cudaStream_t stream_,
                  char*        vshmem,
                  char const*  name_)
        : plan(plan_),
          count(0),
          stream(stream_),
          name(name_),
          grid(plan.grid_size),
          vshmem(vshmem),
          has_shmem((size_t)core::get_max_shared_memory_per_block() >= (size_t)plan.shared_memory_size),
          shmem_size(has_shmem ? plan.shared_memory_size : 0)
    {
      assert(plan.grid_size > 0);
    }

    THRUST_RUNTIME_FUNCTION
    void sync() const
    {
      CubDebug(cub::detail::DebugSyncStream(stream));
    }

    template <class K>
    THRUST_RUNTIME_FUNCTION
    static cuda_optional<int> max_blocks_per_sm_impl(K k, int block_threads)
    {
      int occ;
      cudaError_t status = cub::MaxSmOccupancy(occ, k, block_threads);
      return cuda_optional<int>(status == cudaSuccess ? occ : -1, status);
    }

    template <class... Args>
    THRUST_RUNTIME_FUNCTION
    static cuda_optional<int> get_max_blocks_per_sm(AgentPlan plan)
    {
      using tunings_t = typename Agent::Tunings;
      constexpr auto exec_space = cub::detail::runtime_exec_space;
      using dispatcher_t = cub::detail::ptx_dispatch<tunings_t, exec_space>;

      MaxSmFunctor functor{};
      dispatcher_t::exec(functor,
                         plan.block_threads,
                         cub::detail::type_list<Args...>{});
      return functor.result;
    }

    template <class K>
    THRUST_RUNTIME_FUNCTION
    cuda_optional<int> max_sm_occupancy(K k) const
    {
      return max_blocks_per_sm_impl(k, plan.block_threads);
    }

    template<class K>
    THRUST_RUNTIME_FUNCTION
    void print_info(K k) const
    {
      #if THRUST_DEBUG_SYNC_FLAG
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
      #else
      (void)k;
      #endif
    }

    // If we are guaranteed to have enough shared memory
    // don't compile other kernel which accepts pointer
    // and save on compilations
    template <class PtxPlan, class... Args>
    void THRUST_RUNTIME_FUNCTION
    launch_impl(thrust::detail::true_type, PtxPlan, Args... args) const
    {
      assert(has_shmem && vshmem == NULL);
      auto kernel_ptr = _kernel_agent<Agent, PtxPlan, Args...>;
      print_info(kernel_ptr);
      launcher::triple_chevron(grid, plan.block_threads, shmem_size, stream)
        .doit(kernel_ptr, args...);
    }

    // If there is a risk of not having enough shared memory
    // we compile generic kernel instead.
    // This kernel is likely to be somewhat slower, but it can accomodate
    // both shared and virtualized shared memories.
    // Alternative option is to compile two kernels, one using shared and one
    // using virtualized shared memory. While this can be slightly faster if we
    // do actually have enough shared memory, the compilation time will double.
    //
    template <class PtxPlan, class... Args>
    void THRUST_RUNTIME_FUNCTION
    launch_impl(thrust::detail::false_type, PtxPlan, Args... args) const
    {
      assert((has_shmem && vshmem == NULL) ||
             (!has_shmem && vshmem != NULL && shmem_size == 0));
      auto kernel_ptr = _kernel_agent_vshmem<Agent, PtxPlan, Args...>;
      print_info(kernel_ptr);
      launcher::triple_chevron(grid, plan.block_threads, shmem_size, stream)
        .doit(kernel_ptr, vshmem, args...);
    }

    /// Launches the kernel using the supplied PtxPlan.
    template <typename PtxPlan, typename... Args>
    THRUST_RUNTIME_FUNCTION
    void launch_ptx_plan(PtxPlan, Args &&...args)
    {
      // From CUB commit c4c5d03683049cec8b60cb7781e873dfece43e17:
      // Check if we can use simpler code path that assumes that all shared
      // memory can fit on chip.
      // Otherwise, a kernel will be compiled which can also accept virtualized
      // shared memory, in case there is not enough on chip. This kernel is
      // about 10% slower
      constexpr std::size_t plan_size = temp_storage_size<PtxPlan>::value;
      using has_enough_shmem_t =
        thrust::detail::integral_constant<bool,
                                          (plan_size <= MAX_SHMEM_PER_BLOCK)>;

      launch_impl(has_enough_shmem_t{}, PtxPlan{}, std::forward<Args>(args)...);
      sync();
    }

    /// Uses cub::detail::ptx_dispatch to launch the kernel.
    template <typename... Tunings, typename... Args>
    THRUST_RUNTIME_FUNCTION
    void launch_ptx_dispatch(cub::detail::type_list<Tunings...>, Args &&...args)
    {
      using tunings_t           = cub::detail::type_list<Tunings...>;
      constexpr auto exec_space = cub::detail::runtime_exec_space;
      using dispatcher_t = cub::detail::ptx_dispatch<tunings_t, exec_space>;

      dispatcher_t::exec(LaunchFunctor{*this},
                         std::forward<Args>(args)...);
    }

  private:
    // Dispatch functor for cub::detail::ptx_dispatch that calls
    // `launch_ptx_plan` on the AgentLauncher.
    struct LaunchFunctor
    {
      AgentLauncher& agent_launcher;

      template <typename Tuning, typename... Args>
      THRUST_RUNTIME_FUNCTION
      void operator()(cub::detail::type_wrapper<Tuning>, Args &&...args)
      {
        using ptx_plan_t = typename Agent::template PtxPlan<Tuning>;
        agent_launcher.launch_ptx_plan(ptx_plan_t{},
                                       std::forward<Args>(args)...);
      }
    }; // LaunchFunctor

    // Dispatch functor for cub::detail::ptx_dispatch that returns the maximum
    // SM occupancy for a tuned kernel.
    struct MaxSmFunctor
    {
      cuda_optional<int> result{};

      template <typename Tuning, typename... Args>
      THRUST_RUNTIME_FUNCTION
      void operator()(cub::detail::type_wrapper<Tuning>,
                      int block_threads,
                      cub::detail::type_list<Args...>)
      {
        using ptx_plan_t = typename Agent::template PtxPlan<Tuning>;

        auto kernel = _kernel_agent<Agent, ptx_plan_t, Args...>;
        int  occ{};

        const cudaError_t status =
          cub::MaxSmOccupancy(occ, kernel, block_threads);
        result = cuda_optional<int>(status == cudaSuccess ? occ : -1, status);
      }
    }; // MaxSmFunctor

  }; // AgentLauncher

} // namespace core
} // namespace cuda_cub
THRUST_NAMESPACE_END

#endif
