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
#include <thrust/system/cuda/detail/bulk/detail/config.hpp>

// the purpose of this header is to #include <cuda_runtime_api> without causing
// warnings from redefinitions of __host__ and __device__.
// carefully save their definitions and restore them
// push_macro & pop_macro were introduced to gcc in version 4.3


#if !defined(__GNUC__) || ((10000 * __GNUC__ + 100 * __GNUC_MINOR__ + __GNUC_PATCHLEVEL__) >= 40400) || defined(__clang__)
#  ifdef __host__
#    pragma push_macro("__host__")
#    undef __host__
#    define BULK_HOST_NEEDS_RESTORATION
#  endif
#  ifdef __device__
#    pragma push_macro("__device__")
#    undef __device__
#    define BULK_DEVICE_NEEDS_RESTORATION
#  endif
#endif // __GNUC__


#include <cuda_runtime_api.h>


#if !defined(__GNUC__) || ((10000 * __GNUC__ + 100 * __GNUC_MINOR__ + __GNUC_PATCHLEVEL__) >= 40400) || defined(__clang__)
#  ifdef BULK_HOST_NEEDS_RESTORATION
#    pragma pop_macro("__host__")
#    undef BULK_HOST_NEEDS_RESTORATION
#  endif
#  ifdef BULK_DEVICE_NEEDS_RESTORATION
#    pragma pop_macro("__device__")
#    undef BULK_DEVICE_NEEDS_RESTORATION
#  endif
#endif // __GNUC__

