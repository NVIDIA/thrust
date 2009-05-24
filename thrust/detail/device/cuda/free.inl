/*
 *  Copyright 2008-2009 NVIDIA Corporation
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


/*! \file free.inl
 *  \brief Inline file for free.h.
 */

#include <cuda_runtime_api.h>

namespace thrust
{

namespace detail
{

namespace device
{

namespace cuda
{

void free(thrust::device_ptr<void> ptr)
{
  cudaFree(ptr.get());
} // end free()

} // end namespace cuda

} // end namespace device

} // end namespace detail

} // end namespace thrust


