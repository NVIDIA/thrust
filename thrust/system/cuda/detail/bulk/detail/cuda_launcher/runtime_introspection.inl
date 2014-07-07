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


#if __BULK_HAS_CUDART__


BULK_NAMESPACE_PREFIX
namespace bulk
{
namespace detail
{
namespace runtime_introspection_detail
{


__host__ __device__
inline void get_device_properties(device_properties_t &p, int device_id)
{
  cudaDeviceProp properties;
  
  bulk::detail::throw_on_error(cudaGetDeviceProperties(&properties, device_id), "get_device_properties(): after cudaGetDeviceProperties");

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
} // end get_device_properties()


} // end runtime_introspection_detail


__host__ __device__
inline device_properties_t device_properties(int device_id)
{
  device_properties_t prop;

  cudaError_t error = cudaDeviceGetAttribute(&prop.major,           cudaDevAttrComputeCapabilityMajor,      device_id);
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

  throw_on_error(error, "cudaDeviceGetProperty in get_device_properties");

  return prop;
}


__host__ __device__
inline int current_device()
{
  int result = -1;

  bulk::detail::throw_on_error(cudaGetDevice(&result), "current_device(): after cudaGetDevice");

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
// cudaFuncGetAttributes(), used below, only exists when __CUDACC__ is defined
#ifdef __CUDACC__
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


#endif // __BULK_HAS_CUDART__

