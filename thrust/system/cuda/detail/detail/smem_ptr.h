/*
 *  Copyright 2008-2012 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in ccudaliance with the License.
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

#include <thrust/memory.h>
#include <thrust/system/cuda/detail/tag.h>

namespace thrust
{
namespace system
{
namespace cuda
{
namespace detail
{
namespace detail
{

// XXX CUDA 4.1 WAR
template<typename T>
inline __host__ __device__ void free(T* volatile) {}


template<typename T>
  class smem_ptr
    : public thrust::pointer<
        T,
        thrust::system::cuda::tag,
        T &,
        smem_ptr<T>
      >
{
  typedef thrust::pointer<T,thrust::system::cuda::tag, T&, smem_ptr<T> > super_t;

  public:
    // __shared__ memory is small enough that we only ever need unsigned int
    typedef unsigned int difference_type;

    inline __device__
    smem_ptr() {}

    template<typename OtherT>
    inline __device__
    smem_ptr(OtherT *ptr)
      : super_t(ptr)
    {}
};


template<typename T>
inline __device__ smem_ptr<T> make_smem_ptr(T *ptr)
{
  return smem_ptr<T>(ptr);
}


}
}
} // end cuda
} // end thrust
} // end thrust

