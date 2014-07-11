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

#include <thrust/detail/config.h>
#include <thrust/system/cuda/detail/synchronize.h>
#include <thrust/system/cuda/detail/guarded_cuda_runtime_api.h>
#include <thrust/system/cuda/detail/throw_on_error.h>

namespace thrust
{
namespace system
{
namespace cuda
{
namespace detail
{


inline __host__ __device__
void synchronize(const char *message)
{
  throw_on_error(cudaDeviceSynchronize(), message);
} // end synchronize()


inline __host__ __device__
void synchronize_if_enabled(const char *message)
{
// XXX this could potentially be a runtime decision
//     note we always have to synchronize in __device__ code
#if __THRUST_SYNCHRONOUS || defined(__CUDA_ARCH__)
  synchronize(message);
#else
  // WAR "unused parameter" warning
  (void) message;
#endif
} // end synchronize_if_enabled()


} // end namespace detail
} // end namespace cuda
} // end namespace system
} // end namespace thrust

