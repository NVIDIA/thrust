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
#include <thrust/system/cuda/detail/bulk/detail/terminate.hpp>
#include <thrust/system_error.h>
#include <thrust/system/cuda/error.h>
#include <cstdio>


BULK_NAMESPACE_PREFIX
namespace bulk
{
namespace detail
{


inline __host__ __device__
void throw_on_error(cudaError_t e, const char *message)
{
  if(e)
  {
#ifndef __CUDA_ARCH__
    throw thrust::system_error(e, thrust::cuda_category(), message);
#else
#  if (__BULK_HAS_PRINTF__ && __BULK_HAS_CUDART__)
    printf("Error after %s: %s\n", message, cudaGetErrorString(e));
#  elif __BULK_HAS_PRINTF__
    printf("Error: %s\n", message);
#  endif
    bulk::detail::terminate();
#endif
  } // end if
} // end throw_on_error()


} // end detail
} // end bulk
BULK_NAMESPACE_SUFFIX

