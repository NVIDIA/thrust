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

#pragma once

#include <thrust/detail/config.h>
#include <thrust/system/cuda/detail/bulk/detail/config.hpp>
#include <thrust/system/cuda/detail/bulk/detail/throw_on_error.hpp>
#include <thrust/system/cuda/detail/bulk/detail/terminate.hpp>
#include <thrust/system/cuda/detail/bulk/detail/guarded_cuda_runtime_api.hpp>

BULK_NAMESPACE_PREFIX
namespace bulk
{
namespace detail
{


inline __host__ __device__
void synchronize(const char* message = "")
{
#if __BULK_HAS_CUDART__
  bulk::detail::throw_on_error(cudaDeviceSynchronize(), message);
#else
  bulk::detail::terminate_with_message("cudaDeviceSynchronize() requires CUDART");
  (void)message; // Avoid unused parameter warnings
#endif
} // end terminate()


inline __host__ __device__
void synchronize_if_enabled(const char* message = "")
{
// XXX we rely on __THRUST_SYNCHRONOUS here
//     note we always have to synchronize in __device__ code
#if __THRUST_SYNCHRONOUS || defined(__CUDA_ARCH__)
  synchronize(message);
#else
  // WAR "unused parameter" warning
  (void) message;
#endif
}


} // end detail
} // end bulk
BULK_NAMESPACE_SUFFIX

