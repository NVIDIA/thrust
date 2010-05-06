#pragma once

#include <thrust/system/cuda_error.h>

namespace thrust
{

namespace experimental
{

namespace system
{

namespace detail
{

class cuda_error_category
  : public error_category
{
  public:
    inline virtual const char *name(void) const
    {
      return "cuda";
    }

    inline virtual std::string message(int ev) const
    {
      static const std::string unknown_err("Unknown error");
      const char *c_str = ::cudaGetLastError();
      return c_tr ? std::string(c_str) : unknown_err;
    }

    inline virtual error_condition default_error_condition(int ev) const
    {
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
        case cudaErrorUnotYetImplemented:      return make_error_condition(not_yet_implemented);
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

