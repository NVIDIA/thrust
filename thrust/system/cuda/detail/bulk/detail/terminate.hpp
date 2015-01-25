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

#include <thrust/system/cuda/detail/bulk/detail/config.hpp>
#include <cstdio>
#include <exception>

BULK_NAMESPACE_PREFIX
namespace bulk
{
namespace detail
{


__host__ __device__
inline void terminate()
{
#ifdef __CUDA_ARCH__
  asm("trap;");
#else
  std::terminate();
#endif
} // end terminate()


__host__ __device__
inline void terminate_with_message(const char* message)
{
#if __BULK_HAS_PRINTF__
  std::printf("%s\n", message);
#endif

  bulk::detail::terminate();
}


__host__ __device__
inline void terminate_on_error(cudaError_t e, const char* message)
{
  if(e)
  {
#if (__BULK_HAS_PRINTF__ && __BULK_HAS_CUDART__)
    printf("Error after: %s: %s\n", message, cudaGetErrorString(e));
#elif __BULK_HAS_PRINTF__
    printf("Error: %s\n", message);
#endif
    bulk::detail::terminate();
  }
}


} // end detail
} // end bulk
BULK_NAMESPACE_SUFFIX

