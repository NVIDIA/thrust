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

// do not attempt to compile this code, which relies on 
// CUDART, without system support
#if THRUST_DEVICE_BACKEND == THRUST_DEVICE_BACKEND_CUDA

#include <thrust/detail/device/cuda/no_throw_free.h>
#include <cuda_runtime_api.h>

namespace thrust
{

namespace detail
{

namespace device
{

namespace cuda
{


template<unsigned int DummyParameterToAvoidInstantiation>
  void no_throw_free(thrust::device_ptr<void> ptr) throw()
{
  try
  {
    // ignore the CUDA error if it exists
    cudaFree(ptr.get());
  }
  catch(...)
  {
    ;
  }
} // end no_throw_free()


} // end cuda

} // end device

} // end detail

} // end namespace thrust

#endif // THRUST_DEVICE_BACKEND

