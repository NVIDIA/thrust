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
/*! \file config.h
 *  \brief Defines platform configuration.
 */

#pragma once

#ifdef __CUDACC__

#include <cuda.h>

#if CUDA_VERSION < 2030
#error "CUDA v2.3 or newer is required"
#endif 

// XXX WAR this problem with Snow Leopard + CUDA 2.3a
#if defined(__APPLE__)
#if _GLIBCXX_ATOMIC_BUILTINS == 1
#undef _GLIBCXX_ATOMIC_BUILTINS
#endif // _GLIBCXX_ATOMIC_BUILTINS
#endif // __APPLE__

#else

// if we're not compiling with nvcc,
// #include this to define what __host__ and __device__ mean
// XXX ideally, we wouldn't require an installation of CUDA
#include <host_defines.h>

#endif // __CUDACC__

