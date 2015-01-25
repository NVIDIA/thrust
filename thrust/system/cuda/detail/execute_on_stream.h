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
#include <thrust/system/cuda/detail/execution_policy.h>
#include <thrust/system/cuda/detail/guarded_cuda_runtime_api.h>

namespace thrust
{
namespace system
{
namespace cuda
{
namespace detail
{


// given any old execution_policy, we return stream 0 by default
template<typename DerivedPolicy>
__host__ __device__
inline cudaStream_t stream(const execution_policy<DerivedPolicy> &exec)
{
  return 0;
}


// base class for execute_on_stream
template<typename DerivedPolicy>
class execute_on_stream_base
  : public thrust::system::cuda::detail::execution_policy<DerivedPolicy>
{
  public:
    __host__ __device__
    execute_on_stream_base()
      : m_stream(0)
    {}

    __host__ __device__
    execute_on_stream_base(cudaStream_t stream)
      : m_stream(stream)
    {}

    __host__ __device__
    DerivedPolicy on(const cudaStream_t &s) const
    {
      // create a copy of *this to return
      // make sure it is the derived type
      DerivedPolicy result = thrust::detail::derived_cast(*this);

      // change the result's stream to s
      result.set_stream(s);

      return result;
    }

  private:
    // stream() is a friend function because we call it through ADL
    __host__ __device__
    friend inline cudaStream_t stream(const execute_on_stream_base &exec)
    {
      return exec.m_stream;
    }

    __host__ __device__
    inline void set_stream(const cudaStream_t &s)
    {
      m_stream = s;
    }

    cudaStream_t m_stream;
};


// execution policy which submits kernel launches on a given stream
class execute_on_stream
  : public execute_on_stream_base<execute_on_stream>
{
  typedef execute_on_stream_base<execute_on_stream> super_t;

  public:
    __host__ __device__
    inline execute_on_stream(cudaStream_t stream) 
      : super_t(stream)
    {}
};


} // end detail
} // end cuda
} // end system
} // end thrust

