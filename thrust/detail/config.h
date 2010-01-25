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

#define THRUST_UNKNOWN 0

// XXX reserve 0 for undefined
#define THRUST_CUDA    1
#define THRUST_OMP     2

#ifndef THRUST_DEVICE_BACKEND
#define THRUST_DEVICE_BACKEND THRUST_CUDA
#endif // THRUST_DEVICE_BACKEND

// enumerate compilers we know about
#define THRUST_COMPILER_UNKNOWN 0
#define THRUST_COMPILER_MSVC    1
#define THRUST_COMPILER_GCC     2

// figure out which compiler we're using
#if   defined(_MSC_VER)
#define THRUST_COMPILER THRUST_COMPILER_MSVC
#define THRUST_DEPRECATED __declspec(deprecated)
#elif defined(__GNUC__)
#define THRUST_COMPILER THRUST_COMPILER_GCC
#define THRUST_DEPRECATED __attribute__ ((deprecated)) 
#else
#define THRUST_COMPILER THRUST_COMPILER_UNKNOWN
#define THRUST_DEPRECATED
#endif // THRUST_COMPILER

