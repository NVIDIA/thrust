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
#define THRUST_HOST_COMPILER_CLANG   3

// enumerate device compilers we know about
#define THRUST_DEVICE_COMPILER_UNKNOWN 0
#define THRUST_DEVICE_COMPILER_MSVC    1
#define THRUST_DEVICE_COMPILER_GCC     2
#define THRUST_DEVICE_COMPILER_NVCC    3
#define THRUST_DEVICE_COMPILER_CLANG   4

// figure out which host compiler we're using
// XXX we should move the definition of THRUST_DEPRECATED out of this logic
#if   defined(_MSC_VER)
#define THRUST_HOST_COMPILER THRUST_HOST_COMPILER_MSVC
#define THRUST_DEPRECATED __declspec(deprecated)
#elif defined(__clang__)
#define THRUST_HOST_COMPILER THRUST_HOST_COMPILER_CLANG
#define THRUST_DEPRECATED __attribute__ ((deprecated)) 
#define THRUST_CLANG_VERSION (__clang_major__ * 10000 + __clang_minor__ * 100 + __clang_patchlevel__)
#elif defined(__GNUC__)
#define THRUST_HOST_COMPILER THRUST_HOST_COMPILER_GCC
#define THRUST_DEPRECATED __attribute__ ((deprecated)) 
#define THRUST_GCC_VERSION (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__)
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
#elif THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_CLANG
// CUDA-capable clang should behave similar to NVCC.
#if defined(__CUDA__) && defined(__clang__)
#define THRUST_DEVICE_COMPILER THRUST_DEVICE_COMPILER_CLANG
#elif defined(__CUDA__)
#define THRUST_DEVICE_COMPILER THRUST_DEVICE_COMPILER_NVCC
#endif
#else
#define THRUST_DEVICE_COMPILER THRUST_DEVICE_COMPILER_UNKNOWN
#endif

// is the device compiler capable of compiling omp?
#ifdef _OPENMP
#define THRUST_DEVICE_COMPILER_IS_OMP_CAPABLE THRUST_TRUE
#else
#define THRUST_DEVICE_COMPILER_IS_OMP_CAPABLE THRUST_FALSE
#endif // _OPENMP

// disable specific MSVC warnings
#if (THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_MSVC) && !defined(__CUDA_ARCH__)
#define __THRUST_DISABLE_MSVC_WARNING_BEGIN(x) \
__pragma(warning(push)) \
__pragma(warning(disable : x))
#define __THRUST_DISABLE_MSVC_WARNING_END(x) \
__pragma(warning(pop))
#else
#define __THRUST_DISABLE_MSVC_WARNING_BEGIN(x)
#define __THRUST_DISABLE_MSVC_WARNING_END(x)
#endif
#define __THRUST_DISABLE_MSVC_POSSIBLE_LOSS_OF_DATA_WARNING(x) \
__THRUST_DISABLE_MSVC_WARNING_BEGIN(4244 4267) \
x;\
__THRUST_DISABLE_MSVC_WARNING_END(4244 4267)
#define __THRUST_DISABLE_MSVC_POSSIBLE_LOSS_OF_DATA_WARNING_BEGIN \
__THRUST_DISABLE_MSVC_WARNING_BEGIN(4244 4267)
#define __THRUST_DISABLE_MSVC_POSSIBLE_LOSS_OF_DATA_WARNING_END \
__THRUST_DISABLE_MSVC_WARNING_END(4244 4267)
#define __THRUST_DISABLE_MSVC_FORCING_VALUE_TO_BOOL(x) \
__THRUST_DISABLE_MSVC_WARNING_BEGIN(4800) \
x;\
__THRUST_DISABLE_MSVC_WARNING_END(4800)
#define __THRUST_DISABLE_MSVC_FORCING_VALUE_TO_BOOL_BEGIN \
__THRUST_DISABLE_MSVC_WARNING_BEGIN(4800)
#define __THRUST_DISABLE_MSVC_FORCING_VALUE_TO_BOOL_END \
__THRUST_DISABLE_MSVC_WARNING_END(4800)
