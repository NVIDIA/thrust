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

#if THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_MSVC
#include <intrin.h>
#define __thrust_compiler_fence() _ReadWriteBarrier()
#elif THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_GCC
#define __thrust_compiler_fence() __sync_synchronize()
#elif THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_UNKNOWN
// allow the code to compile without any guarantees
#define __thrust_compiler_fence()
#endif

