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
#include <thrust/system/cuda/detail/synchronize.h>
#include <thrust/system/cuda/detail/guarded_cuda_runtime_api.h>
#include <thrust/system/cuda/error.h>
#include <thrust/system_error.h>

namespace thrust
{
namespace system
{
namespace cuda
{
namespace detail
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
#else
  // WAR "unused parameter" warning
  (void) message;
#endif
} // end synchronize_if_enabled()

} // end namespace detail
} // end namespace cuda
} // end namespace system
} // end namespace thrust

