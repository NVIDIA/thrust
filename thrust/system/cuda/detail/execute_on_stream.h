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
    execute_on_stream_base()
      : m_stream(0)
    {}

    execute_on_stream_base(cudaStream_t stream)
      : m_stream(stream)
    {}

    friend inline cudaStream_t stream(const execute_on_stream_base &exec)
    {
      return exec.m_stream;
    }

  private:
    cudaStream_t m_stream;
};


// execution policy which submits kernel launches on a given stream
class execute_on_stream
  : public execute_on_stream_base<execute_on_stream>
{
  typedef execute_on_stream_base<execute_on_stream> super_t;

  public:
    inline execute_on_stream(cudaStream_t stream) 
      : super_t(stream)
    {}
};


} // end detail
} // end cuda
} // end system
} // end thrust

