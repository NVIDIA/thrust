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

#include <thrust/iterator/iterator_traits.h>

namespace thrust
{
namespace detail
{

// define these in detail for now
struct cuda_device_space_tag : device_space_tag {};
struct omp_device_space_tag : device_space_tag {};

#if   THRUST_DEVICE_BACKEND == THRUST_DEVICE_BACKEND_CUDA
typedef cuda_device_space_tag default_device_space_tag;
#elif THRUST_DEVICE_BACKEND == THRUST_DEVICE_BACKEND_OMP
typedef omp_device_space_tag  default_device_space_tag;
#else
#error Unknown device backend.
#endif // THRUST_DEVICE_BACKEND

} // end namespace detail
} // end namespace thrust

