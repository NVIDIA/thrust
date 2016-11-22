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
#include <thrust/detail/allocator/allocator_traits.h>
#include <thrust/detail/execute_with_allocator.h>
#include <thrust/system/cuda/detail/execution_policy.h>

BEGIN_NS_THRUST
namespace cuda_cub {

__host__ __device__ inline cudaStream_t default_stream()
{
  return cudaStreamLegacy;
}

template <class Derived>
cudaStream_t __host__ __device__ 
get_stream(execution_policy<Derived> &)
{
  return default_stream();
}

template <class Derived>
cudaError_t THRUST_RUNTIME_FUNCTION
synchronize_stream(execution_policy<Derived> &)
{
  cudaDeviceSynchronize();
  return cudaGetLastError();
}


template <class Derived>
struct execute_on_stream_base : execution_policy<Derived>
{
private:
  cudaStream_t stream;

public:
  __host__ __device__
  execute_on_stream_base(cudaStream_t stream_ = default_stream())
      : stream(stream_) {}

  __host__ __device__
      Derived
      on(cudaStream_t const &s) const
  {
    Derived result = derived_cast(*this);
    result.stream  = s;
    return result;
  }

private:
  friend cudaStream_t __host__ __device__
  get_stream(execute_on_stream_base &exec)
  {
    return exec.stream;
  }

  friend cudaError_t THRUST_RUNTIME_FUNCTION
  synchronize_stream(execute_on_stream_base &exec)
  {
#ifdef __CUDA_ARCH__
#ifdef __THRUST_HAS_CUDART__
    THRUST_UNUSED_VAR(exec);
    cudaDeviceSynchronize();
#endif
#else
    cudaStreamSynchronize(exec.stream);
#endif
    return cudaGetLastError();
  }
};

struct execute_on_stream : execute_on_stream_base<execute_on_stream>
{
  typedef execute_on_stream_base<execute_on_stream> base_t;

  __host__ __device__
  execute_on_stream() : base_t(){};
  __host__ __device__
  execute_on_stream(cudaStream_t stream) : base_t(stream){};
};


struct par_t : execution_policy<par_t>
{
  typedef execution_policy<par_t> base_t;

  __device__ __host__
  par_t() : base_t() {}

  template <class Allocator>
  struct enable_alloc
  {
    typedef typename thrust::detail::enable_if<
        thrust::detail::is_allocator<Allocator>::value,
        thrust::detail::execute_with_allocator<Allocator,
                                               execute_on_stream_base> >::type
        type;
  };

  template <class Allocator>
  __host__ __device__ typename enable_alloc<Allocator>::type
  operator()(Allocator &alloc) const
  {
    return thrust::detail::execute_with_allocator<
        Allocator,
        execute_on_stream_base>(alloc);
  }

  execute_on_stream __device__ __host__
  on(cudaStream_t const &stream) const
  {
    return execute_on_stream(stream);
  }
};

#ifdef __CUDA_ARCH__
static const __device__ par_t par;
#else
static const par_t par;
#endif
}    // namespace cuda_

namespace system {
namespace cuda {
  using thrust::cuda_cub::par;
  namespace detail {
    using thrust::cuda_cub::par_t;
  }
} // namesapce cuda
} // namespace system

namespace cuda {
using thrust::cuda_cub::par;
} // namespace cuda

END_NS_THRUST

