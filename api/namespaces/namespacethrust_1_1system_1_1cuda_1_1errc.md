---
title: thrust::system::cuda::errc
nav_exclude: true
has_children: true
has_toc: false
---

# Namespace `thrust::system::cuda::errc`

Namespace for CUDA Runtime errors. 

<code class="doxybook">
<span>namespace thrust::system::cuda::errc {</span>
<br>
<span>enum <b><a href="{{ site.baseurl }}/api/namespaces/namespacethrust_1_1system_1_1cuda_1_1errc.html#enum-errc-t">errc&#95;t</a></b>;</span>
<span>} /* namespace thrust::system::cuda::errc */</span>
</code>

## Types

<h3 id="enum-errc-t">
Enum <code>thrust::system::cuda::errc::errc&#95;t</code>
</h3>

| Enumerator | Value | Description |
|------------|-------|-------------|
| `success` | `cudaSuccess` |  |
| `missing_configuration` | `cudaErrorMissingConfiguration` |  |
| `memory_allocation` | `cudaErrorMemoryAllocation` |  |
| `initialization_error` | `cudaErrorInitializationError` |  |
| `launch_failure` | `cudaErrorLaunchFailure` |  |
| `prior_launch_failure` | `cudaErrorPriorLaunchFailure` |  |
| `launch_timeout` | `cudaErrorLaunchTimeout` |  |
| `launch_out_of_resources` | `cudaErrorLaunchOutOfResources` |  |
| `invalid_device_function` | `cudaErrorInvalidDeviceFunction` |  |
| `invalid_configuration` | `cudaErrorInvalidConfiguration` |  |
| `invalid_device` | `cudaErrorInvalidDevice` |  |
| `invalid_value` | `cudaErrorInvalidValue` |  |
| `invalid_pitch_value` | `cudaErrorInvalidPitchValue` |  |
| `invalid_symbol` | `cudaErrorInvalidSymbol` |  |
| `map_buffer_object_failed` | `cudaErrorMapBufferObjectFailed` |  |
| `unmap_buffer_object_failed` | `cudaErrorUnmapBufferObjectFailed` |  |
| `invalid_host_pointer` | `cudaErrorInvalidHostPointer` |  |
| `invalid_device_pointer` | `cudaErrorInvalidDevicePointer` |  |
| `invalid_texture` | `cudaErrorInvalidTexture` |  |
| `invalid_texture_binding` | `cudaErrorInvalidTextureBinding` |  |
| `invalid_channel_descriptor` | `cudaErrorInvalidChannelDescriptor` |  |
| `invalid_memcpy_direction` | `cudaErrorInvalidMemcpyDirection` |  |
| `address_of_constant_error` | `cudaErrorAddressOfConstant` |  |
| `texture_fetch_failed` | `cudaErrorTextureFetchFailed` |  |
| `texture_not_bound` | `cudaErrorTextureNotBound` |  |
| `synchronization_error` | `cudaErrorSynchronizationError` |  |
| `invalid_filter_setting` | `cudaErrorInvalidFilterSetting` |  |
| `invalid_norm_setting` | `cudaErrorInvalidNormSetting` |  |
| `mixed_device_execution` | `cudaErrorMixedDeviceExecution` |  |
| `cuda_runtime_unloading` | `cudaErrorCudartUnloading` |  |
| `unknown` | `cudaErrorUnknown` |  |
| `not_yet_implemented` | `cudaErrorNotYetImplemented` |  |
| `memory_value_too_large` | `cudaErrorMemoryValueTooLarge` |  |
| `invalid_resource_handle` | `cudaErrorInvalidResourceHandle` |  |
| `not_ready` | `cudaErrorNotReady` |  |
| `insufficient_driver` | `cudaErrorInsufficientDriver` |  |
| `set_on_active_process_error` | `cudaErrorSetOnActiveProcess` |  |
| `no_device` | `cudaErrorNoDevice` |  |
| `ecc_uncorrectable` | `cudaErrorECCUncorrectable` |  |
| `startup_failure` | `cudaErrorStartupFailure` |  |

<code>errc&#95;t</code> enumerates the kinds of CUDA Runtime errors. 


