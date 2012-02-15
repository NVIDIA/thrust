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

#include <thrust/detail/config.h>

// do not attempt to compile this file, which relies on 
// CUDART without system support
#if THRUST_DEVICE_BACKEND == THRUST_DEVICE_BACKEND_CUDA

#include <thrust/detail/backend/cuda/free.h>
#include <cuda_runtime_api.h>
#include <thrust/system_error.h>
#include <thrust/system/cuda_error.h>

namespace thrust
{
namespace detail
{
namespace backend
{
namespace cuda
{

template<unsigned int DummyParameterToAvoidInstantiation>
void free(thrust::device_ptr<void> ptr)
{
  cudaError_t error = cudaFree(ptr.get());

  if(error)
  {
    throw thrust::system_error(error, thrust::cuda_category());
  } // end error
} // end free()

} // end namespace cuda
} // end namespace backend
} // end namespace detail
} // end namespace thrust

#endif // THRUST_DEVICE_BACKEND

