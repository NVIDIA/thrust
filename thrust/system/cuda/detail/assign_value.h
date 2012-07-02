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
#include <thrust/system/cuda/detail/tag.h>
#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/detail/copy.h>

namespace thrust
{
namespace system
{
namespace cuda
{
namespace detail
{


// XXX WAR an issue with msvc 2005 (cl v14.00) which creates multiply-defined
//     symbols resulting from assign_value
#if (THRUST_HOST_COMPILER == THRUST_HOST_COMPILER) && (_MSC_VER <= 1400)

namespace
{

template<typename Pointer1, typename Pointer2>
inline __host__ __device__
  void assign_value_msvc2005_war(Pointer1 dst, Pointer2 src)
{
  // XXX war nvbugs/881631
  struct war_nvbugs_881631
  {
    __host__ inline static void host_path(Pointer1 dst, Pointer2 src)
    {
      thrust::copy(src, src + 1, dst);
    }

    __device__ inline static void device_path(Pointer1 dst, Pointer2 src)
    {
      *thrust::raw_pointer_cast(dst) = *thrust::raw_pointer_cast(src);
    }
  };

#ifndef __CUDA_ARCH__
  war_nvbugs_881631::host_path(dst,src);
#else
  war_nvbugs_881631::device_path(dst,src);
#endif // __CUDA_ARCH__
} // end assign_value_msvc2005_war()

} // end anon namespace

template<typename System, typename Pointer1, typename Pointer2>
inline __host__ __device__
  void assign_value(thrust::system::cuda::detail::dispatchable<System> &s, Pointer1 dst, Pointer2 src)
{
  return assign_value_msvc2005_war(dst,src);
} // end assign_value()

#else

template<typename System, typename Pointer1, typename Pointer2>
inline __host__ __device__
  void assign_value(thrust::system::cuda::detail::dispatchable<System> &, Pointer1 dst, Pointer2 src)
{
  // XXX war nvbugs/881631
  struct war_nvbugs_881631
  {
    __host__ inline static void host_path(Pointer1 dst, Pointer2 src)
    {
      thrust::copy(src, src + 1, dst);
    }

    __device__ inline static void device_path(Pointer1 dst, Pointer2 src)
    {
      *thrust::raw_pointer_cast(dst) = *thrust::raw_pointer_cast(src);
    }
  };

#ifndef __CUDA_ARCH__
  war_nvbugs_881631::host_path(dst,src);
#else
  war_nvbugs_881631::device_path(dst,src);
#endif // __CUDA_ARCH__
} // end assign_value()

#endif // msvc 2005 WAR


template<typename Pointer1, typename Pointer2>
inline __host__ __device__
  void assign_value(cpp_to_cuda, Pointer1 dst, Pointer2 src)
{
#if __CUDA_ARCH__
  cuda::tag t;
  thrust::system::cuda::detail::assign_value(t, dst, src);
#else
  thrust::copy(src, src + 1, dst);
#endif
} // end assign_value()

template<typename Pointer1, typename Pointer2>
inline __host__ __device__
  void assign_value(cuda_to_cpp, Pointer1 dst, Pointer2 src)
{
#if __CUDA_ARCH__
  cuda::tag t;
  thrust::system::cuda::detail::assign_value(t, dst, src);
#else
  thrust::copy(src, src + 1, dst);
#endif
} // end assign_value()

  
} // end detail
} // end cuda
} // end system
} // end thrust

