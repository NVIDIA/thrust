/*
 *  Copyright 2008-2020 NVIDIA Corporation
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

#include <thrust/system/cuda/error.h>
#include <thrust/system/cuda/detail/guarded_cuda_runtime_api.h>
#include <thrust/system/system_error.h>
#include <exception>
#include <cstdio>

namespace thrust
{

namespace system
{


error_code make_error_code(cuda::errc::errc_t e)
{
  return error_code(static_cast<int>(e), cuda_category());
} // end make_error_code()


error_condition make_error_condition(cuda::errc::errc_t e)
{
  return error_condition(static_cast<int>(e), cuda_category());
} // end make_error_condition()


namespace cuda
{

class cuda_error_category
  : public error_category
{
  public:
    inline cuda_error_category(void) {}

    inline virtual const char *name(void) const
    {
      return "cuda";
    }

    inline virtual std::string message(int ev) const
    {
      char const* const unknown_str  = "unknown error";
      char const* const unknown_name = "cudaErrorUnknown";
      char const* c_str  = ::cudaGetErrorString(static_cast<cudaError_t>(ev));
      char const* c_name = ::cudaGetErrorName(static_cast<cudaError_t>(ev));
      return std::string(c_name ? c_name : unknown_name)
           + ": " + (c_str ? c_str : unknown_str);
    }

    inline virtual error_condition default_error_condition(int ev) const
    {
      using namespace cuda::errc;

      if(ev < ::cudaErrorApiFailureBase)
      {
        return make_error_condition(static_cast<errc_t>(ev));
      }

      return system_category().default_error_condition(ev);
    }
};

__host__ __device__ inline void terminate() noexcept
{
  if (THRUST_IS_DEVICE_CODE) {
    #if THRUST_INCLUDE_DEVICE_CODE
      asm("trap;");
    #endif
  } else {
    #if THRUST_INCLUDE_HOST_CODE
      std::terminate();
    #endif
  }
}

__host__  __device__ inline void throw_on_error(cudaError_t status)
{
  #if THRUST_HAS_CUDART
    // Clear the global CUDA error state which may have been set by the last
    // call. Otherwise, errors may "leak" to unrelated kernel launches.
    cudaGetLastError();
  #endif

  if (cudaSuccess != status)
  {
    if (THRUST_IS_HOST_CODE) {
      #if THRUST_INCLUDE_HOST_CODE
        throw system_error(status, cuda_category());
      #endif
    } else {
      #if THRUST_INCLUDE_DEVICE_CODE
        #if THRUST_HAS_CUDART
          printf("Thrust CUDA backend error: %s: %s\n",
                 cudaGetErrorName(status),
                 cudaGetErrorString(status));
        #else
          printf("Thrust CUDA backend error: %d\n",
                 static_cast<int>(status));
        #endif
        terminate();
      #endif
    }
  }
}

__host__ __device__ inline void
throw_on_error(cudaError_t status, char const* msg)
{
#if THRUST_HAS_CUDART
  // Clear the global CUDA error state which may have been set by the last
  // call. Otherwise, errors may "leak" to unrelated kernel launches.
  cudaGetLastError();
#endif

  if (cudaSuccess != status)
  {
    if (THRUST_IS_HOST_CODE) {
      #if THRUST_INCLUDE_HOST_CODE
        throw system_error(status, cuda_category(), msg);
      #endif
    } else {
      #if THRUST_INCLUDE_DEVICE_CODE
        #if THRUST_HAS_CUDART
          printf("Thrust CUDA backend error: %s: %s: %s\n",
                 cudaGetErrorName(status),
                 cudaGetErrorString(status),
                 msg);
        #else
          printf("Thrust CUDA backend error: %d: %s \n",
                 static_cast<int>(status),
                 msg);
        #endif
        terminate();
      #endif
    }
  }
}

} // end namespace cuda

const error_category &cuda_category(void)
{
  static const thrust::system::cuda::cuda_error_category result;
  return result;
}

} // end namespace system

} // end namespace thrust

