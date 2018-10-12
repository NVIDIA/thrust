/*
 *  Copyright 2008-2018 NVIDIA Corporation
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

#include <thrust/detail/config/cpp_dialect.h>

#include <cstddef>

#if THRUST_CPP_DIALECT >= 2011
#  ifndef __has_cpp_attribute
#    define __has_cpp_attribute(X) 0
#  endif

#  if __has_cpp_attribute(nodiscard)
#    define THRUST_NODISCARD [[nodiscard]]
#  elif __has_cpp_attribute(gnu::warn_unused_result)
#    define THRUST_NODISCARD [[gnu::warn_unused_result]]
#  endif

#  define THRUST_CONSTEXPR constexpr
#  define THRUST_OVERRIDE override
#  define THRUST_DEFAULT = default;
#  define THRUST_NOEXCEPT noexcept
#  define THRUST_FINAL final
#  define THRUST_STATIC_CONSTANT static constexpr
#else
#  define THRUST_CONSTEXPR 
#  define THRUST_OVERRIDE
#  define THRUST_DEFAULT {}
#  define THRUST_NOEXCEPT throw()
#  define THRUST_FINAL
#  define THRUST_STATIC_CONSTANT static const
#endif

#ifndef THRUST_NODISCARD
#  define THRUST_NODISCARD
#endif

#ifdef __CUDA_ARCH__
#  if   THRUST_CPP_DIALECT >= 2017
#    define THRUST_INLINE_CONSTANT inline const __device__
#  elif THRUST_CPP_DIALECT >= 2011
#    define THRUST_INLINE_CONSTANT static const __device__
#  else
#    define THRUST_INLINE_CONSTANT static const __device__
#  endif
#else
#  if   THRUST_CPP_DIALECT >= 2017
#    define THRUST_INLINE_CONSTANT inline constexpr
#  elif THRUST_CPP_DIALECT >= 2011
#    define THRUST_INLINE_CONSTANT static constexpr
#  else
#    define THRUST_INLINE_CONSTANT static const
#  endif
#endif

