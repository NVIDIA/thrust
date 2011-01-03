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

#include <thrust/detail/config.h>

// do not attempt to compile this code, which relies on 
// CUDART, without system support
#if THRUST_DEVICE_BACKEND == THRUST_DEVICE_BACKEND_CUDA

#include <thrust/detail/device/cuda/malloc.h>
#include <cuda_runtime_api.h>
#include <thrust/device_ptr.h>
#include <stdexcept>

namespace thrust
{

namespace detail
{

namespace device
{

namespace cuda
{

template<unsigned int DummyParameterToPreventInstantiation>
thrust::device_ptr<void> malloc(const std::size_t n)
{
  void *result = 0;

  cudaError_t error = cudaMalloc(reinterpret_cast<void**>(&result), n);

  if(error)
  {
    throw std::bad_alloc();
  } // end if

  return thrust::device_ptr<void>(result);
} // end malloc()

} // end namespace cuda

} // end namespace device

} // end namespace detail

} // end namespace thrust

#endif // THRUST_DEVICE_BACKEND_CUDA

