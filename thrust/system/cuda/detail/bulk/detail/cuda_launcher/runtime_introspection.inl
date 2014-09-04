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

#include <thrust/system/cuda/detail/bulk/detail/config.hpp>
#include <thrust/system/cuda/detail/bulk/detail/cuda_launcher/runtime_introspection.hpp>
#include <thrust/system/cuda/detail/bulk/detail/throw_on_error.hpp>
#include <thrust/system/cuda/detail/guarded_cuda_runtime_api.h>
#include <thrust/detail/util/blocking.h>
#include <thrust/detail/minmax.h>
#include <thrust/system_error.h>
#include <thrust/system/cuda/error.h>


BULK_NAMESPACE_PREFIX
namespace bulk
{
namespace detail
{


__host__ __device__
inline device_properties_t device_properties_uncached(int device_id)
{
  device_properties_t prop = {0,{0,0,0},0,0,0,0,0,0,0};

  cudaError_t error = cudaErrorNoDevice;

#if __BULK_HAS_CUDART__
  error = cudaDeviceGetAttribute(&prop.major,           cudaDevAttrComputeCapabilityMajor,      device_id);
  error = cudaDeviceGetAttribute(&prop.maxGridSize[0],              cudaDevAttrMaxGridDimX,                 device_id);
  error = cudaDeviceGetAttribute(&prop.maxGridSize[1],              cudaDevAttrMaxGridDimY,                 device_id);
  error = cudaDeviceGetAttribute(&prop.maxGridSize[2],              cudaDevAttrMaxGridDimZ,                 device_id);
  error = cudaDeviceGetAttribute(&prop.maxThreadsPerBlock,          cudaDevAttrMaxThreadsPerBlock,          device_id);
  error = cudaDeviceGetAttribute(&prop.maxThreadsPerMultiProcessor, cudaDevAttrMaxThreadsPerMultiProcessor, device_id);
  error = cudaDeviceGetAttribute(&prop.minor,                       cudaDevAttrComputeCapabilityMinor,      device_id);
  error = cudaDeviceGetAttribute(&prop.multiProcessorCount,         cudaDevAttrMultiProcessorCount,         device_id);
  error = cudaDeviceGetAttribute(&prop.regsPerBlock,                cudaDevAttrMaxRegistersPerBlock,        device_id);
  int temp;
  error = cudaDeviceGetAttribute(&temp,                             cudaDevAttrMaxSharedMemoryPerBlock,     device_id);
  prop.sharedMemPerBlock = temp;
  error = cudaDeviceGetAttribute(&prop.warpSize,                    cudaDevAttrWarpSize,                    device_id);
#else
  (void) device_id; // Surpress unused parameter warnings
#endif

  throw_on_error(error, "cudaDeviceGetProperty in get_device_properties");

  return prop;
}


inline device_properties_t device_properties_cached(int device_id)
{
  // cache the result of get_device_properties, because it is slow
  // only cache the first few devices
  static const int max_num_devices                              = 16;

  static bool properties_exist[max_num_devices]                 = {0};
  static device_properties_t device_properties[max_num_devices] = {};

  if(device_id >= max_num_devices)
  {
    return device_properties_uncached(device_id);
  }

  if(!properties_exist[device_id])
  {
    device_properties[device_id] = device_properties_uncached(device_id);

    // disallow the compiler to move the write to properties_exist[device_id]
    // before the initialization of device_properties[device_id]
    __thrust_compiler_fence();
    
    properties_exist[device_id] = true;
  }

  return device_properties[device_id];
}


__host__ __device__
inline device_properties_t device_properties(int device_id)
{
#ifndef __CUDA_ARCH__
  return device_properties_cached(device_id);
#else
  return device_properties_uncached(device_id);
#endif
}


__host__ __device__
inline int current_device()
{
  int result = -1;

#if __BULK_HAS_CUDART__
  bulk::detail::throw_on_error(cudaGetDevice(&result), "current_device(): after cudaGetDevice");
#endif

  if(result < 0)
  {
    bulk::detail::throw_on_error(cudaErrorNoDevice, "current_device(): after cudaGetDevice"); 
  }

  return result;
}


__host__ __device__
inline device_properties_t device_properties()
{
  return device_properties(current_device());
}


template <typename KernelFunction>
__host__ __device__
inline function_attributes_t function_attributes(KernelFunction kernel)
{
#if __BULK_HAS_CUDART__
  typedef void (*fun_ptr_type)();

  fun_ptr_type fun_ptr = reinterpret_cast<fun_ptr_type>(kernel);

  cudaFuncAttributes attributes;
  
  bulk::detail::throw_on_error(cudaFuncGetAttributes(&attributes, fun_ptr), "function_attributes(): after cudaFuncGetAttributes");

  // be careful about how this is initialized!
  function_attributes_t result = {
    attributes.constSizeBytes,
    attributes.localSizeBytes,
    attributes.maxThreadsPerBlock,
    attributes.numRegs,
    attributes.ptxVersion,
    attributes.sharedSizeBytes
  };

  return result;
#else
  return function_attributes_t();
#endif // __CUDACC__
}

__host__ __device__
inline size_t compute_capability(const device_properties_t &properties)
{
  return 10 * properties.major + properties.minor;
}


__host__ __device__
inline size_t compute_capability()
{
  return compute_capability(device_properties());
}


} // end namespace detail
} // end namespace bulk
BULK_NAMESPACE_SUFFIX

