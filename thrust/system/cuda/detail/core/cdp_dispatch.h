/*
 *  Copyright 2021 NVIDIA Corporation
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

/**
 * \file
 * Utilities for CUDA dynamic parallelism.
 */

#include <thrust/detail/config.h>

#include <cub/detail/cdp_dispatch.cuh>

#ifdef DOXYGEN_SHOULD_SKIP_THIS // Only parse this during doxygen passes:

/**
 * \def THRUST_IGNORE_DEPRECATED_FORK_JOIN_CDP
 *
 * Thrust has deprecated support for device-side launch of synchronous
 * algorithms using the CUDA system.
 *
 * By default, invoking a synchronous Thrust algorithm on the CUDA system from
 * device code will emit a compile-time deprecation warning and fallback to
 * a serial implementation.
 *
 * If this macro is defined, Thrust will revert to the deprecated behavior of
 * launching synchronous CUDA algorithms from the device using fork/join CDP and
 * the compile-time warnings will be suppressed. Note that this behavior may not
 * be supported on all CUDA devices and may be removed in the future.
 *
 * To silence the warning and continue using the sequential implementation,
 * explicitly pass the `thrust::seq` execution policy at the Thrust algorithm
 * callsite.
 */
#define THRUST_IGNORE_DEPRECATED_FORK_JOIN_CDP

/**
 * \def THRUST_CDP_DISPATCH
 *
 * This is a modified version of CUB_CDP_DISPATCH that is aware of Thrust's
 * deprecated fork/join CDP model. See CUB_CDP_DISPATCH's documentation for more
 * details.
 *
 * `par_impl` and `seq_impl` are blocks of C++ statements enclosed in
 * parentheses, similar to NV_IF_TARGET blocks:
 *
 * When invoked from the host, the par_impl is always used, since kernel
 * launches from the host are always supported.
 *
 * When invoked from the device, the seq_impl will always be used, since Thrust
 * is deprecating support for fork/join CDP due to poor performance.
 *
 * To continue using Thrust's fork/join CDP during the deprecation period,
 * define the macro THRUST_IGNORE_DEPRECATED_FORK_JOIN_CDP. This will silence
 * deprecation warnings and restore previous behavior -- Thrust will launch one
 * or more kernels from the device and block until all work is complete.
 *
 * ```
 * THRUST_CDP_DISPATCH((launch_parallel_kernel();), (run_serial_impl();));
 * ```
 */
#define THRUST_CDP_DISPATCH(par_impl, seq_impl)

/**
 * \def THRUST_FORK_JOIN_KERNEL_LAUNCH_EXEC_SPACE
 * The CUDA execution space annotion used when fork/join kernel launches are
 * permitted.
 */
#define THRUST_FORK_JOIN_KERNEL_LAUNCH_EXEC_SPACE

/**
 * \def THRUST_DEPRECATED_FORK_JOIN_CDP
 * C++ attribute used to mark internal Thrust CNP trampoline functions as
 * deprecated when invoked from an NVC++ device pass.
 */
#define THRUST_DEPRECATED_FORK_JOIN_CDP

#else // Doc pass above, impls below

// THRUST_DEPRECATED_FORK_JOIN_CDP ---------------------------------------------
// NVCC device pass and no deprecation opt-out:
#if defined(__CUDACC__) && defined(__CUDA_ARCH__) &&                           \
  !defined(THRUST_IGNORE_DEPRECATED_FORK_JOIN_CDP)

#define THRUST_DEPRECATED_FORK_JOIN_CDP                                        \
  THRUST_DEPRECATED_MSG_IMPL("Device-side launch of Thrust algorithms using "  \
                             "CUDA Dynamic Parallelism is deprecated. "        \
                             "This operation will now execute in a single "    \
                             "CUDA thread using a serial implementation. "     \
                             "To silence this message and continue using the " \
                             "serial implementation, explicitly pass "         \
                             "`thrust::seq` as the execution policy of all "   \
                             "device-side thrust algorithms. "                 \
                             "To force a deprecated fork/join CDP kernel "     \
                             "launch, ensure that RDC is enabled and define "  \
                             "THRUST_IGNORE_DEPRECATED_FORK_JOIN_CDP.")

#else
#define THRUST_DEPRECATED_FORK_JOIN_CDP
#endif

// THRUST_CDP_DISPATCH ---------------------------------------------------------
#if defined(__CUDACC__) && defined(__CUDA_ARCH__) &&                           \
  !defined(THRUST_IGNORE_DEPRECATED_FORK_JOIN_CDP)

// Helper for THRUST_CDP_DISPATCH WAR
THRUST_NAMESPACE_BEGIN
namespace detail
{
inline __host__ __device__ void cdp_no_op() {}
} // namespace detail
THRUST_NAMESPACE_END

// Default device-pass behavior: always execute sequential
#define THRUST_CDP_DISPATCH(par_impl, seq_impl)                                \
  /* For some bizarre reason, the deprecation warnings aren't emitted */       \
  /* without these nonsense no-op instructions? (gcc / nvcc) */                \
  {                                                                            \
    THRUST_NS_QUALIFIER::detail::cdp_no_op();                                  \
  }                                                                            \
  if (false)                                                                   \
  { /* Without this, the device pass won't compile any kernels. */             \
    /* FIXME Just use: NV_IF_TARGET(NV_ANY_TARGET, par_impl); */               \
    CUB_BLOCK_EXPAND(par_impl)                                                 \
  }                                                                            \
  /* FIXME: Just use: NV_IF_TARGET(NV_ANY_TARGET, seq_impl) */                 \
  do                                                                           \
  {                                                                            \
    CUB_BLOCK_EXPAND(seq_impl)                                                 \
  } while (false)

#else // THRUST_IGNORE_DEPRECATED_FORK_JOIN_CDP

// If the deprecation is ignored or this is a host pass, just use the default
// CUB_CDP_DISPATCH
#define THRUST_CDP_DISPATCH(par_impl, seq_impl)                                \
  CUB_CDP_DISPATCH(par_impl, seq_impl)

#endif // THRUST_IGNORE_DEPRECATED_FORK_JOIN_CDP

// THRUST_FORK_JOIN_KERNEL_LAUNCH_EXEC_SPACE -----------------------------------
#ifndef THRUST_IGNORE_DEPRECATED_FORK_JOIN_CDP
#define THRUST_FORK_JOIN_KERNEL_LAUNCH_EXEC_SPACE __host__
#else
#define THRUST_FORK_JOIN_KERNEL_LAUNCH_EXEC_SPACE CUB_RUNTIME_FUNCTION
#endif

#endif // docs/impl
