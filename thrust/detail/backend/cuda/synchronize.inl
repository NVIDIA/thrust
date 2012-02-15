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

// do not attempt to compile this code, which relies on 
// CUDART, without system support
#if THRUST_DEVICE_BACKEND

#include <thrust/detail/backend/cuda/synchronize.h>
#include <cuda_runtime_api.h>
#include <thrust/system/cuda_error.h>
#include <thrust/system_error.h>

namespace thrust
{

namespace detail
{

namespace backend
{

namespace cuda
{

void synchronize(const char *message)
{
  cudaError_t error = cudaThreadSynchronize();
  if(error)
  {
    throw thrust::system_error(error, thrust::cuda_category(), std::string("synchronize: ") + message);
  } // end if
} // end synchronize()

void synchronize_if_enabled(const char *message)
{
// XXX this could potentially be a runtime decision
#if __THRUST_SYNCHRONOUS
  synchronize(message);
#endif
} // end synchronize_if_enabled()

} // end cuda

} // end backend

} // end detail

} // end thrust

#endif // THRUST_DEVICE_BACKEND

