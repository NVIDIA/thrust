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

#include <thrust/detail/config.h>
#include <thrust/system/cuda/detail/runtime_introspection.h>
#include <thrust/system/cuda/detail/guarded_cuda_runtime_api.h>
#include <thrust/detail/util/blocking.h>
#include <thrust/detail/minmax.h>
#include <thrust/system_error.h>
#include <thrust/system/cuda/error.h>


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


inline void get_device_properties(device_properties_t &p, int device_id)
{
  cudaDeviceProp properties;
  
  cudaError_t error = cudaGetDeviceProperties(&properties, device_id);
  
  if(error)
    throw thrust::system_error(error, thrust::cuda_category());

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


inline device_properties_t device_properties(int device_id)
{
  // cache the result of get_device_properties, because it is slow
  // only cache the first few devices
  static const int max_num_devices                              = 16;

  static bool properties_exist[max_num_devices]                 = {0};
  static device_properties_t device_properties[max_num_devices] = {};

  if(device_id >= max_num_devices)
  {
    device_properties_t result;
    runtime_introspection_detail::get_device_properties(result, device_id);
    return result;
  }

  if(!properties_exist[device_id])
  {
    runtime_introspection_detail::get_device_properties(device_properties[device_id], device_id);

    // disallow the compiler to move the write to properties_exist[device_id]
    // before the initialization of device_properties[device_id]
    __thrust_compiler_fence();
    
    properties_exist[device_id] = true;
  }

  return device_properties[device_id];
}

inline int current_device()
{
  int result = -1;

  cudaError_t error = cudaGetDevice(&result);

  if(error)
    throw thrust::system_error(error, thrust::cuda_category());

  if(result < 0)
    throw thrust::system_error(cudaErrorNoDevice, thrust::cuda_category());

  return result;
}

inline device_properties_t device_properties(void)
{
  return device_properties(current_device());
}

template <typename KernelFunction>
inline function_attributes_t function_attributes(KernelFunction kernel)
{
// cudaFuncGetAttributes(), used below, only exists when __CUDACC__ is defined
#ifdef __CUDACC__
  typedef void (*fun_ptr_type)();

  fun_ptr_type fun_ptr = reinterpret_cast<fun_ptr_type>(kernel);

  cudaFuncAttributes attributes;
  
  cudaError_t error = cudaFuncGetAttributes(&attributes, fun_ptr);
  
  if(error)
  {
    throw thrust::system_error(error, thrust::cuda_category());
  }

  // be careful about how this is initialized!
  function_attributes_t result = {
    attributes.constSizeBytes,
    attributes.localSizeBytes,
    attributes.maxThreadsPerBlock,
    attributes.numRegs,
    attributes.sharedSizeBytes
  };

  return result;
#else
  return function_attributes_t();
#endif // __CUDACC__
}

inline size_t compute_capability(const device_properties_t &properties)
{
  return 10 * properties.major + properties.minor;
}

inline size_t compute_capability(void)
{
  return compute_capability(device_properties());
}


} // end namespace detail
} // end namespace cuda
} // end namespace system
} // end namespace thrust

