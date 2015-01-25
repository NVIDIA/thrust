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

#include <thrust/system/cuda/detail/bulk/detail/config.hpp>
#include <thrust/system/cuda/detail/bulk/detail/guarded_cuda_runtime_api.hpp>
#include <thrust/system/cuda/detail/bulk/detail/throw_on_error.hpp>
#include <thrust/system/cuda/detail/bulk/detail/terminate.hpp>
#include <thrust/detail/swap.h>
#include <cstring>


BULK_NAMESPACE_PREFIX
namespace bulk
{
namespace detail
{


// this thing has ownership semantics like unique_ptr, so copy and assign are more like moves
template<typename T>
class parameter_ptr
{
  public:
    typedef T element_type;

    __host__ __device__
    explicit parameter_ptr(element_type *ptr)
      : m_ptr(ptr)
    {}

    // XXX copy emulates a move
    __host__ __device__
    parameter_ptr(const parameter_ptr& other_)
    {
      parameter_ptr& other = const_cast<parameter_ptr&>(other_);
      thrust::swap(m_ptr, other.m_ptr);
    }

    __host__ __device__
    ~parameter_ptr()
    {
#if __BULK_HAS_CUDART__
      if(m_ptr)
      {
        bulk::detail::terminate_on_error(cudaFree(m_ptr), "in parameter_ptr dtor");
      }
#else
      bulk::detail::terminate_with_message("parameter_ptr dtor: cudaFree requires CUDART");
#endif
    }

    // XXX assign emulates a move
    __host__ __device__
    parameter_ptr& operator=(const parameter_ptr& other_)
    {
      parameter_ptr& other = const_cast<parameter_ptr&>(other_);
      thrust::swap(m_ptr, other.m_ptr);
      return *this;
    }

    __host__ __device__
    T* get() const
    {
      return m_ptr;
    }

  private:
    T *m_ptr;
};


template<typename T>
__host__ __device__
parameter_ptr<T> make_parameter(const T& x)
{
  T* raw_ptr = 0;

  // allocate
#if __BULK_HAS_CUDART__
  bulk::detail::throw_on_error(cudaMalloc(&raw_ptr, sizeof(T)), "make_parameter(): after cudaMalloc");
#else
  bulk::detail::terminate_with_message("make_parameter(): cudaMalloc requires CUDART\n");
#endif

  // do a trivial copy
#ifndef __CUDA_ARCH__
  bulk::detail::throw_on_error(cudaMemcpy(raw_ptr, &x, sizeof(T), cudaMemcpyHostToDevice),
                               "make_parameter(): after cudaMemcpy");
#else
  std::memcpy(raw_ptr, &x, sizeof(T));
#endif

  return parameter_ptr<T>(raw_ptr);
}


} // end detail
} // end bulk
BULK_NAMESPACE_SUFFIX

