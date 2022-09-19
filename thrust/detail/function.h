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
#include <thrust/detail/raw_reference_cast.h>

THRUST_NAMESPACE_BEGIN

namespace detail
{

template <typename Function, typename Result>
struct wrapped_function
{
  // mutable because Function::operator() might be const
  mutable Function m_f;

  inline THRUST_HOST_DEVICE
  wrapped_function()
      : m_f()
  {}

  inline THRUST_HOST_DEVICE
  wrapped_function(const Function& f)
      : m_f(f)
  {}

  __thrust_exec_check_disable__
  template <typename Argument>
  __thrust_forceinline__ THRUST_HOST_DEVICE
  Result operator()(Argument& x) const
  {
    return static_cast<Result>(m_f(thrust::raw_reference_cast(x)));
  }

  __thrust_exec_check_disable__
  template <typename Argument>
  __thrust_forceinline__ THRUST_HOST_DEVICE
  Result operator()(const Argument& x) const
  {
    return static_cast<Result>(m_f(thrust::raw_reference_cast(x)));
  }

  __thrust_exec_check_disable__
  template <typename Argument1, typename Argument2>
  __thrust_forceinline__ THRUST_HOST_DEVICE
  Result operator()(Argument1& x, Argument2& y) const
  {
    return static_cast<Result>(m_f(thrust::raw_reference_cast(x),
                                   thrust::raw_reference_cast(y)));
  }

  __thrust_exec_check_disable__
  template <typename Argument1, typename Argument2>
  __thrust_forceinline__ THRUST_HOST_DEVICE
  Result operator()(const Argument1& x, Argument2& y) const
  {
    return static_cast<Result>(m_f(thrust::raw_reference_cast(x),
                                   thrust::raw_reference_cast(y)));
  }

  __thrust_exec_check_disable__
  template <typename Argument1, typename Argument2>
  __thrust_forceinline__ THRUST_HOST_DEVICE
  Result operator()(const Argument1& x, const Argument2& y) const
  {
    return static_cast<Result>(m_f(thrust::raw_reference_cast(x),
                                   thrust::raw_reference_cast(y)));
  }

  __thrust_exec_check_disable__
  template <typename Argument1, typename Argument2>
  __thrust_forceinline__ THRUST_HOST_DEVICE
  Result operator()(Argument1& x, const Argument2& y) const
  {
    return static_cast<Result>(m_f(thrust::raw_reference_cast(x),
                                   thrust::raw_reference_cast(y)));
  }
}; // end wrapped_function

// Specialize for void return types:
template <typename Function>
struct wrapped_function<Function, void>
{
  // mutable because Function::operator() might be const
  mutable Function m_f;
  inline THRUST_HOST_DEVICE
  wrapped_function()
    : m_f()
  {}

  inline THRUST_HOST_DEVICE
  wrapped_function(const Function& f)
    : m_f(f)
  {}

  __thrust_exec_check_disable__
  template <typename Argument>
  __thrust_forceinline__ THRUST_HOST_DEVICE
  void operator()(Argument& x) const
  {
    m_f(thrust::raw_reference_cast(x));
  }

  __thrust_exec_check_disable__
  template <typename Argument>
  __thrust_forceinline__ THRUST_HOST_DEVICE
  void operator()(const Argument& x) const
  {
    m_f(thrust::raw_reference_cast(x));
  }

  __thrust_exec_check_disable__
  template <typename Argument1, typename Argument2>
  __thrust_forceinline__ THRUST_HOST_DEVICE
  void operator()(Argument1& x, Argument2& y) const
  {
    m_f(thrust::raw_reference_cast(x), thrust::raw_reference_cast(y));
  }

  __thrust_exec_check_disable__
  template <typename Argument1, typename Argument2>
  __thrust_forceinline__ THRUST_HOST_DEVICE
  void operator()(const Argument1& x, Argument2& y) const
  {
    m_f(thrust::raw_reference_cast(x), thrust::raw_reference_cast(y));
  }
  __thrust_exec_check_disable__
  template <typename Argument1, typename Argument2>
  __thrust_forceinline__ THRUST_HOST_DEVICE
  void operator()(const Argument1& x, const Argument2& y) const
  {
    m_f(thrust::raw_reference_cast(x), thrust::raw_reference_cast(y));
  }
  __thrust_exec_check_disable__
  template <typename Argument1, typename Argument2>
  __thrust_forceinline__ THRUST_HOST_DEVICE
  void operator()(Argument1& x, const Argument2& y) const
  {
    m_f(thrust::raw_reference_cast(x), thrust::raw_reference_cast(y));
  }
}; // end wrapped_function

} // namespace detail

THRUST_NAMESPACE_END
