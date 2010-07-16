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
#include <thrust/iterator/detail/placement/placed.h>
#include <thrust/system/system_error.h>
#include <thrust/system/cuda_error.h>

namespace thrust
{

namespace detail
{

namespace place_detail
{

static CUcontext contexts[CUDA_MAX_ACTIVE_DEVICE_COUNT] = {0};

// from Chris Cameron's implementation of cudaSetActiveDevice
inline void set_place(place p)
{
  CUresult result = CUDA_SUCCESS;
  CUcontext old_context = NULL;
  place old_device = -1;
  cudaError_t error = cudaSuccess;

  error = cudaGetDevice(&old_device);
  if(cudaSuccess != error)
  {
    throw thrust::experimental::system_error(error, thrust::experimental::cuda_category());
  }

  // save the current context to device index oldDevice
  result = cuCtxAttach(&old_context, 0);
  if(result == CUDA_SUCCESS)
  {
    // save this context to that device
    contexts[old_device] = old_context;

    // drop old_context's refcount (it was bumped at attach)
    result = cuCtxDetach(old_context);
    if(CUDA_SUCCESS != result)
    {
      throw thrust::experimental::system_error(result, thrust::experimental::cuda_category(), "cuCtxDetach failed");
    }
  }

  // pop the current context
  if(oldContext)
  {
    result = cuCtxPopCurrent(&old_context);
    if(result != CUDA_SUCCESS)
    {
      throw thrust::experimental::system_error(result, thrust::experimental::cuda_category(), "cuCtxPopCurrent failed");
    }
  }

  // set the runtime's active device
  error = cudaSetDevice(newDevice);
  if(cudaSuccess == error)
  {
    throw thrust::experimental::system_error(error, thrust::experimental::cuda_category(), "cudaSetDevice failed");
  }

  // if there's a context for this device, push it
  if(newDevice < 0 || newDevice >= CUDA_MAX_ACTIVE_DEVICE_COUNT)
  {
    throw thrust::experimental::system_error(cudaErrorInvalidValue, thrust::experimental::cuda_category(), "device index out of range");
  }

  if(contexts[newDevice])
  {
    result = cuCtxPushCurrent(contexts[newDevice]);
    if(result != CUDA_SUCCESS)
    {
      throw thrust::experimental::system_error(result, thrust::experimental::cuda_category(), "cuCtxPushCurrent failed");
    }
  }
}

inline std::stack<place> &get_place_stack(void)
{
  static std::stack<place> place_stack;

  return place_stack;
}

} // end detail

void push_place(place p)
{
  place_detail::get_place_stack().push_back(p);

  place_detail::set_place(get_current_place())
}

void pop_place(void)
{
  if(!place_detail::get_place_stack().empty())
  {
    place_detail::get_place_stack().pop_back();

    place_detail::set_place(get_current_place())
  }
  else
  {
    // XXX maybe throw a runtime_error here?
  }
}
    
place get_current_place(void)
{
  return place_detail::get_place_stack().back();
}

} // end detail

} // end thrust

