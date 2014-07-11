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

BULK_NAMESPACE_PREFIX
namespace bulk
{
namespace detail
{


inline __device__ unsigned int __isShared(const void *ptr)
{
  // XXX WAR unused variable warning
  (void) ptr;

  unsigned int ret;

#if __CUDA_ARCH__ >= 200
  asm volatile ("{ \n\t"
                "    .reg .pred p; \n\t"
                "    isspacep.shared p, %1; \n\t"
                "    selp.u32 %0, 1, 0, p;  \n\t"
#  if (defined(_MSC_VER) && defined(_WIN64)) || defined(__LP64__)
                "} \n\t" : "=r"(ret) : "l"(ptr));
#  else
                "} \n\t" : "=r"(ret) : "r"(ptr));
#  endif
#else
  ret = 0;
#endif

  return ret;
} // end __isShared()


inline __device__ bool is_shared(const void *ptr)
{
  return __isShared(ptr);
} // end is_shared()


inline __device__ bool is_global(const void *ptr)
{
  // XXX WAR unused variable warning
  (void) ptr;

#if __CUDA_ARCH__ >= 200
  return __isGlobal(ptr);
#else
  return false;
#endif
} // end is_global()


} // end detail
} // end bulk
BULK_NAMESPACE_SUFFIX

