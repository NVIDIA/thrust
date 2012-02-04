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


/*! \file thrust/system/cuda/error.h
 *  \brief CUDA-specific error reporting
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/type_traits.h>
#include <thrust/system/error_code.h>
#include <driver_types.h>

namespace thrust
{

namespace system
{

namespace cuda
{

/*! \addtogroup system
 *  \{
 */

// To construct an error_code after a CUDA Runtime error:
//
//   error_code(::cudaGetLastError(), cuda_category())

// XXX N3000 prefers enum class errc { ... }
namespace errc
{

/*! \p errc_t enumerates the kinds of CUDA Runtime errors.
 */
enum errc_t
{
  // from cuda/include/driver_types.h
  // mirror their order
  success                      = cudaSuccess,
  missing_configuration        = cudaErrorMissingConfiguration,
  memory_allocation            = cudaErrorMemoryAllocation,
  initialization_error         = cudaErrorInitializationError,
  launch_failure               = cudaErrorLaunchFailure,
  prior_launch_failure         = cudaErrorPriorLaunchFailure,
  launch_timeout               = cudaErrorLaunchTimeout,
  launch_out_of_resources      = cudaErrorLaunchOutOfResources,
  invalid_device_function      = cudaErrorInvalidDeviceFunction,
  invalid_configuration        = cudaErrorInvalidConfiguration,
  invalid_device               = cudaErrorInvalidDevice,
  invalid_value                = cudaErrorInvalidValue,
  invalid_pitch_value          = cudaErrorInvalidPitchValue,
  invalid_symbol               = cudaErrorInvalidSymbol,
  map_buffer_object_failed     = cudaErrorMapBufferObjectFailed,
  unmap_buffer_object_failed   = cudaErrorUnmapBufferObjectFailed,
  invalid_host_pointer         = cudaErrorInvalidHostPointer,
  invalid_device_pointer       = cudaErrorInvalidDevicePointer,
  invalid_texture              = cudaErrorInvalidTexture,
  invalid_texture_binding      = cudaErrorInvalidTextureBinding,
  invalid_channel_descriptor   = cudaErrorInvalidChannelDescriptor,
  invalid_memcpy_direction     = cudaErrorInvalidMemcpyDirection,
  address_of_constant_error    = cudaErrorAddressOfConstant,
  texture_fetch_failed         = cudaErrorTextureFetchFailed,
  texture_not_bound            = cudaErrorTextureNotBound,
  synchronization_error        = cudaErrorSynchronizationError,
  invalid_filter_setting       = cudaErrorInvalidFilterSetting,
  invalid_norm_setting         = cudaErrorInvalidNormSetting,
  mixed_device_execution       = cudaErrorMixedDeviceExecution,
  cuda_runtime_unloading       = cudaErrorCudartUnloading,
  unknown                      = cudaErrorUnknown,
  not_yet_implemented          = cudaErrorNotYetImplemented,
  memory_value_too_large       = cudaErrorMemoryValueTooLarge,
  invalid_resource_handle      = cudaErrorInvalidResourceHandle,
  not_ready                    = cudaErrorNotReady,
  insufficient_driver          = cudaErrorInsufficientDriver,
  set_on_active_process_error  = cudaErrorSetOnActiveProcess,
  no_device                    = cudaErrorNoDevice,
  ecc_uncorrectable            = cudaErrorECCUncorrectable,
  startup_failure              = cudaErrorStartupFailure
}; // end errc_t


} // end namespace errc

} // end namespace cuda

/*! \return A reference to an object of a type derived from class \p thrust::error_category.
 *  \note The object's \p equivalent virtual functions shall behave as specified
 *        for the class \p thrust::error_category. The object's \p name virtual function shall
 *        return a pointer to the string <tt>"cuda"</tt>. The object's
 *        \p default_error_condition virtual function shall behave as follows:
 *
 *        If the argument <tt>ev</tt> corresponds to a CUDA error value, the function
 *        shall return <tt>error_condition(ev,cuda_category())</tt>.
 *        Otherwise, the function shall return <tt>system_category.default_error_condition(ev)</tt>.
 */
inline const error_category &cuda_category(void);


// XXX N3000 prefers is_error_code_enum<cuda::errc>

/*! Specialization of \p is_error_code_enum for \p cuda::errc::errc_t
 */
template<> struct is_error_code_enum<cuda::errc::errc_t> : thrust::detail::true_type {};


// XXX replace cuda::errc::errc_t with cuda::errc upon c++0x
/*! \return <tt>error_code(static_cast<int>(e), cuda::error_category())</tt>
 */
inline error_code make_error_code(cuda::errc::errc_t e);


// XXX replace cuda::errc::errc_t with cuda::errc upon c++0x
/*! \return <tt>error_condition(static_cast<int>(e), cuda::error_category())</tt>.
 */
inline error_condition make_error_condition(cuda::errc::errc_t e);

/*! \} // end system
 */


} // end system

namespace cuda
{

// XXX replace with using system::cuda_errc upon c++0x
namespace errc = system::cuda::errc;

} // end cuda

using system::cuda_category;

} // end namespace thrust

#include <thrust/system/cuda/detail/error.inl>

