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

#include <thrust/system/cuda_error.h>

namespace thrust
{

namespace experimental
{

namespace system
{


error_code make_error_code(cuda_errc::cuda_errc_t e)
{
  return error_code(static_cast<int>(e), cuda_category());
} // end make_error_code()


error_condition make_error_condition(cuda_errc::cuda_errc_t e)
{
  return error_condition(static_cast<int>(e), cuda_category());
} // end make_error_condition()


namespace detail
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
      static const std::string unknown_err("Unknown error");
      const char *c_str = ::cudaGetErrorString(static_cast<cudaError_t>(ev));
      return c_str ? std::string(c_str) : unknown_err;
    }

    inline virtual error_condition default_error_condition(int ev) const
    {
      using namespace cuda_errc;

      switch(ev)
      {
        case cudaSuccess:                      return make_error_condition(success);
        case cudaErrorMissingConfiguration:    return make_error_condition(missing_configuration);
        case cudaErrorMemoryAllocation:        return make_error_condition(memory_allocation);
        case cudaErrorInitializationError:     return make_error_condition(initialization_error);
        case cudaErrorLaunchFailure:           return make_error_condition(launch_failure);
        case cudaErrorPriorLaunchFailure:      return make_error_condition(prior_launch_failure);
        case cudaErrorLaunchTimeout:           return make_error_condition(launch_timeout);
        case cudaErrorLaunchOutOfResources:    return make_error_condition(launch_out_of_resources);
        case cudaErrorInvalidDeviceFunction:   return make_error_condition(invalid_device_function);
        case cudaErrorInvalidConfiguration:    return make_error_condition(invalid_configuration);
        case cudaErrorInvalidDevice:           return make_error_condition(invalid_device);
        case cudaErrorInvalidValue:            return make_error_condition(invalid_value);
        case cudaErrorInvalidPitchValue:       return make_error_condition(invalid_pitch_value);
        case cudaErrorInvalidSymbol:           return make_error_condition(invalid_symbol);
        case cudaErrorMapBufferObjectFailed:   return make_error_condition(map_buffer_object_failed);
        case cudaErrorUnmapBufferObjectFailed: return make_error_condition(unmap_buffer_object_failed);
        case cudaErrorInvalidHostPointer:      return make_error_condition(invalid_host_pointer);
        case cudaErrorInvalidDevicePointer:    return make_error_condition(invalid_device_pointer);
        case cudaErrorInvalidTexture:          return make_error_condition(invalid_texture);
        case cudaErrorInvalidTextureBinding:   return make_error_condition(invalid_texture_binding);
        case cudaErrorInvalidChannelDescriptor:return make_error_condition(invalid_channel_descriptor);
        case cudaErrorInvalidMemcpyDirection:  return make_error_condition(invalid_memcpy_direction);
        case cudaErrorAddressOfConstant:       return make_error_condition(address_of_constant_error);
        case cudaErrorTextureFetchFailed:      return make_error_condition(texture_fetch_failed);
        case cudaErrorTextureNotBound:         return make_error_condition(texture_not_bound);
        case cudaErrorSynchronizationError:    return make_error_condition(synchronization_error);
        case cudaErrorInvalidFilterSetting:    return make_error_condition(invalid_filter_setting);
        case cudaErrorInvalidNormSetting:      return make_error_condition(invalid_norm_setting);
        case cudaErrorMixedDeviceExecution:    return make_error_condition(mixed_device_execution);
        case cudaErrorCudartUnloading:         return make_error_condition(cuda_runtime_unloading);
        case cudaErrorUnknown:                 return make_error_condition(unknown);
        case cudaErrorNotYetImplemented:       return make_error_condition(not_yet_implemented);
        case cudaErrorMemoryValueTooLarge:     return make_error_condition(memory_value_too_large);
        case cudaErrorInvalidResourceHandle:   return make_error_condition(invalid_resource_handle);
        case cudaErrorNotReady:                return make_error_condition(not_ready);
        case cudaErrorInsufficientDriver:      return make_error_condition(cuda_runtime_is_newer_than_driver);
        case cudaErrorSetOnActiveProcess:      return make_error_condition(set_on_active_process_error);
        case cudaErrorNoDevice:                return make_error_condition(no_device);
        case cudaErrorECCUncorrectable:        return make_error_condition(ecc_uncorrectable);
        case cudaErrorStartupFailure:          return make_error_condition(startup_failure);
        case cudaErrorApiFailureBase:          return make_error_condition(api_failure_base);
        default: return system_category().default_error_condition(ev);
      }
    }
}; // end cuda_error_category

} // end detail


const error_category &cuda_category(void)
{
  static const detail::cuda_error_category result;
  return result;
}

} // end namespace system

} // end namespace experimental

} // end namespace thrust

