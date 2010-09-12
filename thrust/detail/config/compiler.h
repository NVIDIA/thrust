/*
 *  Copyright 2008-2010 NVIDIA Corporation
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

/*! \file compiler.h
 *  \brief Compiler-specific configuration
 */

#pragma once

#ifdef __CUDACC__

#include <cuda.h>

// Thrust supports CUDA >= 3.0
#if CUDA_VERSION < 3000
#error "CUDA v3.0 or newer is required"
#endif // CUDA_VERSION

#endif // __CUDACC__

// enumerate host compilers we know about
#define THRUST_HOST_COMPILER_UNKNOWN 0
#define THRUST_HOST_COMPILER_MSVC    1
#define THRUST_HOST_COMPILER_GCC     2

// enumerate host compilers we know about
#define THRUST_DEVICE_COMPILER_UNKNOWN 0
#define THRUST_DEVICE_COMPILER_MSVC    1
#define THRUST_DEVICE_COMPILER_GCC     2
#define THRUST_DEVICE_COMPILER_NVCC    3

// figure out which host compiler we're using
// XXX we should move the definition of THRUST_DEPRECATED out of this logic
#if   defined(_MSC_VER)
#define THRUST_HOST_COMPILER THRUST_HOST_COMPILER_MSVC
#define THRUST_DEPRECATED __declspec(deprecated)
#elif defined(__GNUC__)
#define THRUST_HOST_COMPILER THRUST_HOST_COMPILER_GCC
#define THRUST_DEPRECATED __attribute__ ((deprecated)) 
#else
#define THRUST_HOST_COMPILER THRUST_HOST_COMPILER_UNKNOWN
#define THRUST_DEPRECATED
#endif // THRUST_HOST_COMPILER

// figure out which device compiler we're using
#if defined(__CUDACC__)
#define THRUST_DEVICE_COMPILER THRUST_DEVICE_COMPILER_NVCC
#elif THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_MSVC
#define THRUST_DEVICE_COMPILER THRUST_DEVICE_COMPILER_MSVC
#elif THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_GCC
#define THRUST_DEVICE_COMPILER THRUST_DEVICE_COMPILER_GCC
#else
#define THRUST_DEVICE_COMPILER THRUST_DEVICE_COMPILER_UNKNOWN
#endif

// is the device compiler capable of compiling omp?
#ifdef _OPENMP
#define THRUST_DEVICE_COMPILER_IS_OMP_CAPABLE THRUST_TRUE
#else
#define THRUST_DEVICE_COMPILER_IS_OMP_CAPABLE THRUST_FALSE
#endif // _OPENMP

