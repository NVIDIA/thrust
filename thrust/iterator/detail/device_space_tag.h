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

#if   THRUST_DEVICE_BACKEND == THRUST_DEVICE_BACKEND_CUDA
#include <thrust/system/cuda/detail/tag.h>

namespace thrust
{

typedef thrust::system::cuda::tag device_space_tag;

} // end thrust

#elif  THRUST_DEVICE_BACKEND == THRUST_DEVICE_BACKEND_OMP
#include <thrust/system/omp/detail/tag.h>

namespace thrust
{

typedef thrust::system::omp::tag device_space_tag;

} // end thrust

#elif THRUST_DEVICE_BACKEND == THRUST_DEVICE_BACKEND_TBB
#include <thrust/system/tbb/detail/tag.h>

namespace thrust
{

typedef thrust::system::tbb::tag device_space_tag;

} // end thrust

#else
#error Unknown device backend.
#endif // THRUST_DEVICE_BACKEND

