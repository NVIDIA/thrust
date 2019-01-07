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

/*! Allocates an area of memory available to Thrust's <tt>cuda</tt> system.
 *  \param n Number of bytes to allocate.
 *  \return A <tt>cuda::pointer<void></tt> pointing to the beginning of the newly
 *          allocated memory. A null <tt>cuda::pointer<void></tt> is returned if
 *          an error occurs.
 *  \note The <tt>cuda::pointer<void></tt> returned by this function must be
 *        deallocated with \p cuda::free.
 *  \see cuda::free
 *  \see std::malloc
 */
inline __host__ __device__ pointer<void> malloc(std::size_t n);

/*! Allocates a typed area of memory available to Thrust's <tt>cuda</tt> system.
 *  \param n Number of elements to allocate.
 *  \return A <tt>cuda::pointer<T></tt> pointing to the beginning of the newly
 *          allocated elements. A null <tt>cuda::pointer<T></tt> is returned if
 *          an error occurs.
 *  \note The <tt>cuda::pointer<T></tt> returned by this function must be
 *        deallocated with \p cuda::free.
 *  \see cuda::free
 *  \see std::malloc
 */
template <typename T>
inline __host__ __device__ pointer<T> malloc(std::size_t n);

/*! Deallocates an area of memory previously allocated by <tt>cuda::malloc</tt>.
 *  \param ptr A <tt>cuda::pointer<void></tt> pointing to the beginning of an area
 *         of memory previously allocated with <tt>cuda::malloc</tt>.
 *  \see cuda::malloc
 *  \see std::free
 */
inline __host__ __device__ void free(pointer<void> ptr);

// XXX upon c++11
// template<typename T>
// using allocator = thrust::mr::stateless_resource_allocator<T, memory_resource>;

/*! \p cuda::allocator is the default allocator used by the \p cuda system's containers such as
 *  <tt>cuda::vector</tt> if no user-specified allocator is provided. \p cuda::allocator allocates
 *  (deallocates) storage with \p cuda::malloc (\p cuda::free).
 */
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
  /*! The \p rebind metafunction provides the type of an \p allocator
   *  instantiated with another type.
   *
   *  \tparam U The other type to use for instantiation.
   */
  template <typename U>
  struct rebind
  {
    /*! The typedef \p other gives the type of the rebound \p allocator.
     */
    typedef allocator<U> other;
  };

  /*! No-argument constructor has no effect.
   */
  __host__ __device__
  inline allocator() {}

  /*! Copy constructor has no effect.
   */
  __host__ __device__
 inline allocator(const allocator & other) : base(other) {}

  /*! Constructor from other \p allocator has no effect.
   */
  template <typename U>
  __host__ __device__
  inline allocator(const allocator<U> & other) : base(other) {}

  /*! Destructor has no effect.
   */
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
} // namespace system

namespace cuda {
using thrust::cuda_cub::malloc;
using thrust::cuda_cub::free;
using thrust::cuda_cub::allocator;
}    // end cuda

THRUST_END_NS

#include <thrust/system/cuda/detail/memory.inl>

