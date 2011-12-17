/*
 *  Copyright 2008-2011 NVIDIA Corporation
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
#include <thrust/detail/type_traits/result_of.h>
#include <thrust/detail/raw_reference_cast.h>

namespace thrust
{
namespace detail
{


template<typename Function,
         typename Reference,
         typename Result = typename thrust::detail::result_of<
           Function(typename thrust::detail::raw_reference<Reference>::type)
         >::type
        >
  struct host_device_wrapped_unary_function
{
  // mutable because Function::operator() might be const
  mutable Function m_f;

  __host__ __device__
  host_device_wrapped_unary_function()
    : m_f()
  {}

  __host__ __device__
  host_device_wrapped_unary_function(const Function &f)
    : m_f(f)
  {}

  __host__ __device__ __thrust_forceinline__
  Result operator()(Reference ref) const
  {
    // we static_cast to Result to handle void Result without error
    // in case Function's result is non-void
    return static_cast<Result>(m_f(thrust::raw_reference_cast(ref)));
  }
}; // host_device_wrapped_unary_function


template<typename Function,
         typename Reference,
         typename Result = typename thrust::detail::result_of<
           Function(typename thrust::detail::raw_reference<Reference>::type)
         >::type
        >
  struct host_wrapped_unary_function
{
  // mutable because Function::operator() might be const
  mutable Function m_f;

  host_wrapped_unary_function()
    : m_f()
  {}

  host_wrapped_unary_function(const Function &f)
    : m_f(f)
  {}

  Result operator()(Reference ref) const
  {
    // we static_cast to Result to handle void Result without error
    // in case Function's result is non-void
    return static_cast<Result>(m_f(thrust::raw_reference_cast(ref)));
  }
}; // host_wrapped_unary_function


} // end detail
} // end thrust

