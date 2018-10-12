/*
 *  Copyright 2008-2018 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in ccudaliance with the License.
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

/*! \file thrust/system/cuda/memory.h
 *  \brief Managing memory associated with Thrust's CUDA system.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/system/cuda/memory_resource.h>
#include <thrust/memory.h>
#include <thrust/detail/type_traits.h>
#include <thrust/mr/allocator.h>
#include <ostream>

THRUST_BEGIN_NS
namespace cuda_cub {

inline __host__ __device__ pointer<void> malloc(std::size_t n);

template <typename T>
inline __host__ __device__ pointer<T> malloc(std::size_t n);

inline __host__ __device__ void free(pointer<void> ptr);

// XXX upon c++11
// template<typename T>
// using allocator = thrust::mr::stateless_resource_allocator<T, memory_resource>;
//
template <typename T>
struct allocator
    : thrust::mr::stateless_resource_allocator<
        T,
        system::cuda::memory_resource
    >
{
private:
    typedef thrust::mr::stateless_resource_allocator<
        T,
        system::cuda::memory_resource
    > base;

public:
  template <typename U>
  struct rebind
  {
    typedef allocator<U> other;
  };

  __host__ __device__
  inline allocator() {}

  __host__ __device__
 inline allocator(const allocator & other) : base(other) {}

  template <typename U>
  __host__ __device__
  inline allocator(const allocator<U> & other) : base(other) {}

  __host__ __device__
  inline ~allocator() {}
};    // struct allocator

}    // namespace cuda_cub

namespace system {
namespace cuda {
using thrust::cuda_cub::malloc;
using thrust::cuda_cub::free;
using thrust::cuda_cub::allocator;
} // namespace cuda
} /// namespace system

namespace cuda {
using thrust::cuda_cub::malloc;
using thrust::cuda_cub::free;
using thrust::cuda_cub::allocator;
}    // end cuda

THRUST_END_NS

#include <thrust/system/cuda/detail/memory.inl>

