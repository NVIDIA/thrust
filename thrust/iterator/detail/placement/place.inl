/*
 *  Copyright 2008-2010 NVIDIA Corporation
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

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <thrust/iterator/detail/placement/place.h>
#include <thrust/system/system_error.h>
#include <thrust/system/cuda_error.h>
#include <stack>

namespace thrust
{

namespace detail
{

namespace place_detail
{

const size_t max_active_cuda_device_count = 64;
static CUcontext contexts[max_active_cuda_device_count] = {0};

struct place_core_access
{
  // from Chris Cameron's implementation of cudaSetActiveDevice
  static inline void set_place(place<thrust::detail::cuda_device_space_tag> p)
  {
    CUresult result = CUDA_SUCCESS;
    CUcontext old_context = NULL;
    place<thrust::detail::cuda_device_space_tag> old_device;
    cudaError_t error = cudaSuccess;
  
    error = cudaGetDevice(&old_device.m_resource);
    if(cudaSuccess != error)
    {
      throw thrust::experimental::system_error(error, thrust::experimental::cuda_category(), "set_place(): cudaGetDevice failed");
    }
  
    // save the current context to device index oldDevice
    result = cuCtxAttach(&old_context, 0);
    if(result == CUDA_SUCCESS)
    {
      // save this context to that device
      contexts[old_device.m_resource] = old_context;
  
      // drop old_context's refcount (it was bumped at attach)
      result = cuCtxDetach(old_context);
      if(CUDA_SUCCESS != result)
      {
        throw thrust::experimental::system_error(result, thrust::experimental::cuda_category(), "set_place(): cuCtxDetach failed");
      }
    }
  
    // pop the current context
    if(old_context)
    {
      result = cuCtxPopCurrent(&old_context);
      if(result != CUDA_SUCCESS)
      {
        throw thrust::experimental::system_error(result, thrust::experimental::cuda_category(), "set_place(): cuCtxPopCurrent failed");
      }
    }
  
    // set the runtime's active device
    error = cudaSetDevice(p.m_resource);
    if(error)
    {
      throw thrust::experimental::system_error(error, thrust::experimental::cuda_category(), "set_place(): cudaSetDevice failed");
    }
  
    // if there's a context for this device, push it
    if(p.m_resource < 0 || p.m_resource >= max_active_cuda_device_count)
    {
      throw thrust::experimental::system_error(cudaErrorInvalidValue, thrust::experimental::cuda_category(), "set_place(): device index out of range");
    }
  
    if(contexts[p.m_resource])
    {
      result = cuCtxPushCurrent(contexts[p.m_resource]);
      if(result != CUDA_SUCCESS)
      {
        throw thrust::experimental::system_error(result, thrust::experimental::cuda_category(), "set_place(): cuCtxPushCurrent failed");
      }
    }
  }
}; // end place_core_access

template<typename Space>
inline std::stack<place<Space> > &get_place_stack(void)
{
  static std::stack<place<Space> > place_stack;

  return place_stack;
}

} // end detail

void push_place(place p)
{
  place_detail::get_place_stack<typename place::space>().push(p);

  place_detail::place_core_access::set_place(get_current_place());
}

void pop_place(void)
{
  if(!place_detail::get_place_stack<typename place::space>().empty())
  {
    place_detail::get_place_stack<typename place::space>().pop();

    if(!place_detail::get_place_stack<typename place::space>().empty())
    {
      place_detail::place_core_access::set_place(get_current_place());
    }
  }
  else
  {
    // XXX maybe throw a runtime_error here?
  }
}
    
place get_current_place(void)
{
  return place_detail::get_place_stack<typename place::space>().top();
}

size_t num_places(void)
{
  int result = 0;
  cudaError_t error = cudaGetDeviceCount(&result);
  if(error)
  {
    throw thrust::experimental::system_error(error, thrust::experimental::cuda_category(), "num_places(): cudaGetDeviceCount failed");
  }

  return static_cast<size_t>(result);
}

} // end detail

} // end thrust

