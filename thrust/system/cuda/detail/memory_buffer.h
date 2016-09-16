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

#include <thrust/detail/pointer.h>
#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/pair.h>
#include <thrust/system/cuda/config.h>
#include <thrust/system/cuda/detail/execution_policy.h>

BEGIN_NS_THRUST

// XXX forward declare thrust::get/return_temporary_buffer
// to avoid circular include dependency from thrust/memory.h
//
template<typename T, typename DerivedPolicy>
__host__ __device__
thrust::pair<thrust::pointer<T,DerivedPolicy>, typename thrust::pointer<T,DerivedPolicy>::difference_type>
get_temporary_buffer(const thrust::detail::execution_policy_base<DerivedPolicy> &system, typename thrust::pointer<T,DerivedPolicy>::difference_type n);

template<typename DerivedPolicy, typename Pointer>
__host__ __device__
void return_temporary_buffer(const thrust::detail::execution_policy_base<DerivedPolicy> &system, Pointer p);

namespace cuda_cub {

template <class Derived>
__host__ __device__ void *
get_memory_buffer(execution_policy<Derived> &policy, std::ptrdiff_t n)
{
  return (void *)thrust::raw_pointer_cast(
      thrust::get_temporary_buffer<char>(policy, n).first);
}

template <class Derived>
void __host__ __device__
return_memory_buffer(execution_policy<Derived> &policy, void* ptr)
{
  thrust::return_temporary_buffer(policy,ptr);
}

}    // namespace cuda_cub
END_NS_THRUST

// include thrust/memory.h  after
// we define get/return_memory_buffer
// 
//#include <thrust/memory.h>

#endif

