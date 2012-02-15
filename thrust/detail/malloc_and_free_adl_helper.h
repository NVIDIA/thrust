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

// the purpose of this header is to #include the malloc_and_free.h header
// of the host and device systems. It should be #included in any
// code which uses adl to dispatch malloc or free.

#if   THRUST_HOST_SYSTEM == THRUST_HOST_SYSTEM_CPP
#include <thrust/system/cpp/detail/malloc_and_free.h>
#elif THRUST_HOST_SYSTEM == THRUST_HOST_SYSTEM_OMP
// omp has no malloc_and_free.h
#elif THRUST_HOST_SYSTEM == THRUST_HOST_SYSTEM_TBB
// tbb has no malloc_and_free.h
#else
#error "Unknown host system."
#endif // THRUST_HOST_SYSTEM


#if   THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
#include <thrust/system/cuda/detail/malloc_and_free.h>
#elif THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_OMP
// omp has no malloc_and_free.h
#elif THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_TBB
// tbb has no malloc_and_free.h
#else
#error "Unknown device system."
#endif // THRUST_DEVICE_SYSTEM

