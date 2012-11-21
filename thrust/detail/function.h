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
#include <thrust/detail/raw_reference_cast.h>

namespace thrust
{
namespace detail
{


template<typename Function, typename Result>
  struct host_function
{
  // mutable because Function::operator() might be const
  mutable Function m_f;

  inline host_function()
    : m_f()
  {}

  inline host_function(const Function &f)
    : m_f(f)
  {}

  template<typename Argument>
    inline Result operator()(Argument &x) const
  {
    // we static cast to Result to handle void Result without error
    // in case Function's result is non-void
    return static_cast<Result>(m_f(thrust::raw_reference_cast(x)));
  }

  template<typename Argument>
    inline Result operator()(const Argument &x) const
  {
    // we static cast to Result to handle void Result without error
    // in case Function's result is non-void
    return static_cast<Result>(m_f(thrust::raw_reference_cast(x)));
  }

  template<typename Argument1, typename Argument2>
    inline Result operator()(Argument1 &x, Argument2 &y) const
  {
    // we static cast to Result to handle void Result without error
    // in case Function's result is non-void
    return static_cast<Result>(m_f(thrust::raw_reference_cast(x), thrust::raw_reference_cast(y)));
  }

  template<typename Argument1, typename Argument2>
    inline Result operator()(const Argument1 &x, Argument2 &y) const
  {
    // we static cast to Result to handle void Result without error
    // in case Function's result is non-void
    return static_cast<Result>(m_f(thrust::raw_reference_cast(x), thrust::raw_reference_cast(y)));
  }

  template<typename Argument1, typename Argument2>
    inline Result operator()(const Argument1 &x, const Argument2 &y) const
  {
    // we static cast to Result to handle void Result without error
    // in case Function's result is non-void
    return static_cast<Result>(m_f(thrust::raw_reference_cast(x), thrust::raw_reference_cast(y)));
  }

  template<typename Argument1, typename Argument2>
    inline Result operator()(Argument1 &x, const Argument2 &y) const
  {
    // we static cast to Result to handle void Result without error
    // in case Function's result is non-void
    return static_cast<Result>(m_f(thrust::raw_reference_cast(x), thrust::raw_reference_cast(y)));
  }
}; // end host_function


template<typename Function, typename Result>
  struct device_function
{
  // mutable because Function::operator() might be const
  mutable Function m_f;

  inline __device__ device_function()
    : m_f()
  {}

  inline __device__ device_function(const Function &f)
    : m_f(f)
  {}

  template<typename Argument>
    inline __device__ Result operator()(Argument &x) const
  {
    // we static cast to Result to handle void Result without error
    // in case Function's result is non-void
    return static_cast<Result>(m_f(thrust::raw_reference_cast(x)));
  }

  template<typename Argument>
    inline __device__ Result operator()(const Argument &x) const
  {
    // we static cast to Result to handle void Result without error
    // in case Function's result is non-void
    return static_cast<Result>(m_f(thrust::raw_reference_cast(x)));
  }

  template<typename Argument1, typename Argument2>
    inline __device__ Result operator()(Argument1 &x, Argument2 &y) const
  {
    // we static cast to Result to handle void Result without error
    // in case Function's result is non-void
    return static_cast<Result>(m_f(thrust::raw_reference_cast(x), thrust::raw_reference_cast(y)));
  }

  template<typename Argument1, typename Argument2>
    inline __device__ Result operator()(const Argument1 &x, Argument2 &y) const
  {
    // we static cast to Result to handle void Result without error
    // in case Function's result is non-void
    return static_cast<Result>(m_f(thrust::raw_reference_cast(x), thrust::raw_reference_cast(y)));
  }

  template<typename Argument1, typename Argument2>
    inline __device__ Result operator()(const Argument1 &x, const Argument2 &y) const
  {
    // we static cast to Result to handle void Result without error
    // in case Function's result is non-void
    return static_cast<Result>(m_f(thrust::raw_reference_cast(x), thrust::raw_reference_cast(y)));
  }

  template<typename Argument1, typename Argument2>
    inline __device__ Result operator()(Argument1 &x, const Argument2 &y) const
  {
    // we static cast to Result to handle void Result without error
    // in case Function's result is non-void
    return static_cast<Result>(m_f(thrust::raw_reference_cast(x), thrust::raw_reference_cast(y)));
  }
}; // end device_function


template<typename Function, typename Result>
  struct host_device_function
{
  // mutable because Function::operator() might be const
  mutable Function m_f;

  inline __host__ __device__
  host_device_function()
    : m_f()
  {}

  inline __host__ __device__
  host_device_function(const Function &f)
    : m_f(f)
  {}

  __thrust_hd_warning_disable__
  template<typename Argument>
  inline __host__ __device__
    Result operator()(Argument &x) const
  {
    // we static cast to Result to handle void Result without error
    // in case Function's result is non-void
    return static_cast<Result>(m_f(thrust::raw_reference_cast(x)));
  }

  template<typename Argument>
    inline __host__ __device__ Result operator()(const Argument &x) const
  {
    // we static cast to Result to handle void Result without error
    // in case Function's result is non-void
    return static_cast<Result>(m_f(thrust::raw_reference_cast(x)));
  }

  template<typename Argument1, typename Argument2>
    inline __host__ __device__ Result operator()(Argument1 &x, Argument2 &y) const
  {
    // we static cast to Result to handle void Result without error
    // in case Function's result is non-void
    return static_cast<Result>(m_f(thrust::raw_reference_cast(x), thrust::raw_reference_cast(y)));
  }

  template<typename Argument1, typename Argument2>
    inline __host__ __device__ Result operator()(const Argument1 &x, Argument2 &y) const
  {
    // we static cast to Result to handle void Result without error
    // in case Function's result is non-void
    return static_cast<Result>(m_f(thrust::raw_reference_cast(x), thrust::raw_reference_cast(y)));
  }

  template<typename Argument1, typename Argument2>
    inline __host__ __device__ Result operator()(const Argument1 &x, const Argument2 &y) const
  {
    // we static cast to Result to handle void Result without error
    // in case Function's result is non-void
    return static_cast<Result>(m_f(thrust::raw_reference_cast(x), thrust::raw_reference_cast(y)));
  }

  template<typename Argument1, typename Argument2>
    inline __host__ __device__ Result operator()(Argument1 &x, const Argument2 &y) const
  {
    // we static cast to Result to handle void Result without error
    // in case Function's result is non-void
    return static_cast<Result>(m_f(thrust::raw_reference_cast(x), thrust::raw_reference_cast(y)));
  }
}; // end host_device_function


} // end detail
} // end thrust

