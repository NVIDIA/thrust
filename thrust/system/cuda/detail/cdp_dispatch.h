/*
*  Copyright 2021-2022 NVIDIA Corporation
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

/**
 * \file
 * Utilities for CUDA dynamic parallelism.
 */

#pragma once

#include <cub/config.cuh>
#include <cub/detail/detect_cuda_runtime.cuh>

#include <nv/target>

/**
 * \def THRUST_CDP_DISPATCH
 *
 * If CUDA Dynamic Parallelism / CUDA Nested Parallelism is available, always
 * run the parallel implementation. Otherwise, run the parallel implementation
 * when called from the host, and fallback to the sequential implementation on
 * the device.
 *
 * `par_impl` and `seq_impl` are blocks of C++ statements enclosed in
 * parentheses, similar to NV_IF_TARGET blocks:
 *
 * \code
 * THRUST_CDP_DISPATCH((launch_parallel_kernel();), (run_serial_impl();));
 * \endcode
 */

#ifdef THRUST_RDC_ENABLED

// seq_impl unused.
#define THRUST_CDP_DISPATCH(par_impl, seq_impl)                                \
  NV_IF_TARGET(NV_ANY_TARGET, par_impl, par_impl)

#else // THRUST_RDC_ENABLED

// Special case for NVCC -- need to inform the device path about the kernels
// that are launched from the host path.
#if defined(__CUDACC__) && defined(__CUDA_ARCH__)

// Device-side launch not supported, fallback to sequential in device code.
#define THRUST_CDP_DISPATCH(par_impl, seq_impl)                                \
  if (false)                                                                   \
  { /* Without this, the device pass won't compile any kernels. */             \
    NV_IF_TARGET(NV_ANY_TARGET, par_impl);                                     \
  }                                                                            \
  NV_IF_TARGET(NV_IS_HOST, par_impl, seq_impl)

#else // NVCC device pass

#define THRUST_CDP_DISPATCH(par_impl, seq_impl)                                \
  NV_IF_TARGET(NV_IS_HOST, par_impl, seq_impl)

#endif // NVCC device pass

#endif // THRUST_RDC_ENABLED
