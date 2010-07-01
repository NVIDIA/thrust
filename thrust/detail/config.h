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
/*! \file config.h
 *  \brief Defines platform configuration.
 */

#pragma once

#ifdef __CUDACC__

#include <cuda.h>

// Thrust supports CUDA 2.3 and 3.0
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

#define THRUST_UNKNOWN 0
#define THRUST_FALSE   0
#define THRUST_TRUE    1

// XXX reserve 0 for undefined
#define THRUST_DEVICE_BACKEND_CUDA    1
#define THRUST_DEVICE_BACKEND_OMP     2

#ifndef THRUST_DEVICE_BACKEND
#define THRUST_DEVICE_BACKEND THRUST_DEVICE_BACKEND_CUDA
#endif // THRUST_DEVICE_BACKEND

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

// check for nvcc < 3.0, which strips omp #pragmas
#if (THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC) && (CUDA_VERSION < 3000)
#undef THRUST_DEVICE_COMPILER_IS_OMP_CAPABLE
#define THRUST_DEVICE_COMPILER_IS_OMP_CAPABLE THRUST_FALSE
#endif // (THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC) && (CUDA_VERSION < 3000)

// XXX nvcc < 3.0 strips #pragma, but also crashes upon THRUST_STATIC_ASSERT
// #error out here if the user asks for OMP + nvcc 2.3
#if (THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC) && (CUDA_VERSION < 3000) && (THRUST_DEVICE_BACKEND == THRUST_DEVICE_BACKEND_OMP)
#error "This version of nvcc does not support OpenMP compilation.  Please upgrade to CUDA 3.0."
#endif // nvcc 2.3 OMP guard

#if defined(__DEVICE_EMULATION__)
#if THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_MSVC
#pragma message("-----------------------------------------------------------------------")
#pragma message("| WARNING: Thrust does not support device emulation                    ")
#pragma message("-----------------------------------------------------------------------")
#else
#warning -----------------------------------------------------------------------
#warning | WARNING: Thrust does not support device emulation                    
#warning -----------------------------------------------------------------------
#endif // THRUST_HOST_COMPILER_MSVC
#endif // __DEVICE_EMULATION__

// deprecate nvcc 2.3
#if CUDA_VERSION == 2030
#if THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_MSVC
#pragma message("-----------------------------------------------------------------------")
#pragma message("| WARNING: This version of Thrust does not support CUDA 2.3                            ")
#pragma message("-----------------------------------------------------------------------")
#else
#warning -----------------------------------------------------------------------
#warning | WARNING: This version of Thrust does not support CUDA 2.3
#warning -----------------------------------------------------------------------
#endif // THRUST_HOST_COMPILER_MSVC
#endif 

