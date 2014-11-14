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

#include <thrust/detail/config.h>
#include <thrust/system/cuda/detail/runtime_introspection.h>
#include <thrust/system/cuda/detail/guarded_cuda_runtime_api.h>
#include <thrust/detail/util/blocking.h>
#include <thrust/detail/minmax.h>
#include <thrust/system/cuda/detail/throw_on_error.h>
#include <cstdio>


namespace thrust
{
namespace system
{
namespace cuda
{
namespace detail
{
namespace runtime_introspection_detail
{


__host__ __device__
inline void uncached_device_properties(device_properties_t &p, int device_id)
{
#ifndef __CUDA_ARCH__
  cudaDeviceProp properties;
  
  cudaError_t error = cudaGetDeviceProperties(&properties, device_id);
  
  throw_on_error(error, "cudaGetDeviceProperties in get_device_properties");

  // be careful about how this is initialized!
  device_properties_t temp = {
    properties.major,
    {
      properties.maxGridSize[0],
      properties.maxGridSize[1],
      properties.maxGridSize[2]
    },
    properties.maxThreadsPerBlock,
    properties.maxThreadsPerMultiProcessor,
    properties.minor,
    properties.multiProcessorCount,
    properties.regsPerBlock,
    properties.sharedMemPerBlock,
    properties.warpSize
  };

  p = temp;
#elif (__CUDA_ARCH__ >= 350)
  cudaError_t error = cudaDeviceGetAttribute(&p.major,           cudaDevAttrComputeCapabilityMajor,      device_id);
  error = cudaDeviceGetAttribute(&p.maxGridSize[0],              cudaDevAttrMaxGridDimX,                 device_id);
  error = cudaDeviceGetAttribute(&p.maxGridSize[1],              cudaDevAttrMaxGridDimY,                 device_id);
  error = cudaDeviceGetAttribute(&p.maxGridSize[2],              cudaDevAttrMaxGridDimZ,                 device_id);
  error = cudaDeviceGetAttribute(&p.maxThreadsPerBlock,          cudaDevAttrMaxThreadsPerBlock,          device_id);
  error = cudaDeviceGetAttribute(&p.maxThreadsPerMultiProcessor, cudaDevAttrMaxThreadsPerMultiProcessor, device_id);
  error = cudaDeviceGetAttribute(&p.minor,                       cudaDevAttrComputeCapabilityMinor,      device_id);
  error = cudaDeviceGetAttribute(&p.multiProcessorCount,         cudaDevAttrMultiProcessorCount,         device_id);
  error = cudaDeviceGetAttribute(&p.regsPerBlock,                cudaDevAttrMaxRegistersPerBlock,        device_id);
  int temp;
  error = cudaDeviceGetAttribute(&temp,                          cudaDevAttrMaxSharedMemoryPerBlock,     device_id);
  p.sharedMemPerBlock = temp;
  error = cudaDeviceGetAttribute(&p.warpSize,                    cudaDevAttrWarpSize,                    device_id);

  throw_on_error(error, "cudaDeviceGetProperty in get_device_properties");
#else
  // dunno how we can safely error here.
#endif
} // end get_device_properties()


inline void cached_device_properties(device_properties_t &p, int device_id)
{
  // cache the result of get_device_properties, because it is slow
  // only cache the first few devices
  static const int max_num_devices                              = 16;

  static bool properties_exist[max_num_devices]                 = {0};
  static device_properties_t device_properties[max_num_devices] = {};

  if(device_id >= max_num_devices)
  {
    uncached_device_properties(p, device_id);
  }

  if(!properties_exist[device_id])
  {
    uncached_device_properties(device_properties[device_id], device_id);

    // disallow the compiler to move the write to properties_exist[device_id]
    // before the initialization of device_properties[device_id]
    __thrust_compiler_fence();
    
    properties_exist[device_id] = true;
  }

  p = device_properties[device_id];
}


} // end runtime_introspection_detail


inline __host__ __device__
device_properties_t device_properties(int device_id)
{
  device_properties_t result;
#ifndef __CUDA_ARCH__
  runtime_introspection_detail::cached_device_properties(result, device_id);
#else
  runtime_introspection_detail::uncached_device_properties(result, device_id);
#endif
  return result;
}


inline __host__ __device__
int current_device()
{
  int result = -1;

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 350
  cudaError_t error = cudaGetDevice(&result);

  throw_on_error(error, "cudaGetDevice in current_device");

  if(result < 0)
  {
    throw_on_error(cudaErrorNoDevice, "cudaGetDevice in current_device");
  }
#else
  // dunno how to safely error here
#endif

  return result;
}


inline __host__ __device__
device_properties_t device_properties()
{
  return device_properties(current_device());
}


template<typename KernelFunction>
__host__ __device__
inline function_attributes_t function_attributes(KernelFunction kernel)
{
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 350)
  cudaFuncAttributes attributes;

  typedef void (*fun_ptr_type)();

  fun_ptr_type fun_ptr = reinterpret_cast<fun_ptr_type>(kernel);
  throw_on_error(cudaFuncGetAttributes(&attributes, reinterpret_cast<void*>(fun_ptr)), "cudaFuncGetAttributes in function_attributes");

  // be careful about how this is initialized!
  function_attributes_t result = {
    attributes.constSizeBytes,
    attributes.localSizeBytes,
    attributes.maxThreadsPerBlock,
    attributes.numRegs,
    attributes.ptxVersion,
    attributes.sharedSizeBytes
  };
#else
  function_attributes_t result = {0};
#endif

  return result;
}


inline __host__ __device__
size_t compute_capability(const device_properties_t &properties)
{
  return 10 * properties.major + properties.minor;
}


inline __host__ __device__
size_t compute_capability(void)
{
  return compute_capability(device_properties());
}


} // end namespace detail
} // end namespace cuda
} // end namespace system
} // end namespace thrust

