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

#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/swap.h>

namespace thrust
{
namespace system
{
namespace cuda
{
namespace detail
{


template<typename Pointer1, typename Pointer2>
inline __host__ __device__
void iter_swap(tag, Pointer1 a, Pointer2 b)
{
  // XXX war nvbugs/881631
  struct war_nvbugs_881631
  {
    __host__ inline static void host_path(Pointer1 a, Pointer2 b)
    {
      thrust::swap_ranges(a, a + 1, b);
    }

    __device__ inline static void device_path(Pointer1 a, Pointer2 b)
    {
      using thrust::swap;
      swap(*thrust::raw_pointer_cast(a),
           *thrust::raw_pointer_cast(b));
    }
  };

#ifndef __CUDA_ARCH__
  return war_nvbugs_881631::host_path(a,b);
#else
  return war_nvbugs_881631::device_path(a,b);
#endif // __CUDA_ARCH__
} // end iter_swap()


} // end detail
} // end cuda
} // end system
} // end thrust

