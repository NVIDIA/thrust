/*
 *  Copyright 2008-2011 NVIDIA Corporation
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

namespace thrust
{

namespace detail
{

namespace device
{

namespace cuda
{

template<typename T>
  class extern_shared_ptr
{
// don't attempt to compile with any compiler other than nvcc
// due to use of __shared__ below
#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC
  public:
    __device__
    inline operator T * (void)
    {
      extern __shared__ int4 smem[];
      return reinterpret_cast<T*>(smem);
    }

    __device__
    inline operator const T * (void) const
    {
      extern __shared__ int4 smem[];
      return reinterpret_cast<const T*>(smem);
    }
#endif // THRUST_DEVICE_COMPILER_NVCC
}; // end extern_shared_ptr

} // end cuda

} // end device

} // end detail

} // end thrust

