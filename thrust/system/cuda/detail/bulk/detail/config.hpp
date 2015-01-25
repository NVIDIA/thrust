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

#pragma once

#ifndef BULK_NAMESPACE_PREFIX
#define BULK_NAMESPACE_PREFIX
#endif

#ifndef BULK_NAMESPACE_SUFFIX
#define BULK_NAMESPACE_SUFFIX
#endif

#if defined(__CUDACC__)
#  ifndef __bulk_hd_warning_disable__
#    define __bulk_hd_warning_disable__ \
#    pragma hd_warning_disable
#  endif // __bulk_hd_warning_disable__
#else
#  define __bulk_hd_warning_disable__
#endif // __bulk_hd_warning_disable__

#include <thrust/version.h>

#if THRUST_VERSION < 100800
#error "Bulk requires Thrust v1.8 (http://thrust.github.io) or better."
#endif


#if defined(__CUDACC__)
#  if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__>= 350 && defined(__CUDACC_RDC__))
#    define __BULK_HAS_CUDART__ 1
#  else
#    define __BULK_HAS_CUDART__ 0
#  endif
#else
#  define __BULK_HAS_CUDART__ 0
#endif

#if defined(__CUDACC__)
#  if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 200)
#    define __BULK_HAS_PRINTF__ 1
#  else
#    define __BULK_HAS_PRINTF__ 0
#  endif
#else
#  define __BULK_HAS_PRINTF__ 1
#endif

