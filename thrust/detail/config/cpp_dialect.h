/*
 *  Copyright 2020 NVIDIA Corporation
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

/*! \file cpp_dialect.h
 *  \brief Detect the version of the C++ standard used by the compiler.
 */

#pragma once

#include <thrust/detail/config/deprecated.h>
#include <thrust/version.h> // for THRUST_..._NS macros

#ifndef THRUST_CPP_DIALECT

// MSVC prior to 2015U3 does not expose the C++ dialect and is not supported.
// This is a hard error.
#  ifndef THRUST_IGNORE_DEPRECATED_CPP_DIALECT
#    if defined(_MSC_FULL_VER) && _MSC_FULL_VER < 190024210
#      error "MSVC < 2015 Update 3 is not supported by Thrust."
#    endif
#  endif

// MSVC does not define __cplusplus correctly. _MSVC_LANG is used instead
// (MSVC 2015U3+ only)
#  ifdef _MSVC_LANG
#    define THRUST___CPLUSPLUS _MSVC_LANG
#  else
#    define THRUST___CPLUSPLUS __cplusplus
#  endif

// Detect current standard:
#  if THRUST___CPLUSPLUS < 201103L
#    define THRUST_CPP_DIALECT 2003
#  elif THRUST___CPLUSPLUS < 201402L
#    define THRUST_CPP_DIALECT 2011
#  elif THRUST___CPLUSPLUS < 201703L
#    define THRUST_CPP_DIALECT 2014
#  elif THRUST___CPLUSPLUS == 201703L
#    define THRUST_CPP_DIALECT 2017
#  elif THRUST___CPLUSPLUS > 201703L // unknown, but is higher than 2017.
#    define THRUST_CPP_DIALECT 2020
#  endif

#  undef THRUST___CPLUSPLUS // cleanup

#endif // THRUST_CPP_DIALECT

THRUST_BEGIN_NS

namespace config
{

// Warn for deprecated dialects. Unused function, only compiled so that
// deprecated classes are used.
inline void CheckCppDialect()
{
  // Users may define this to silent the deprecated dialect warnings:
#ifndef THRUST_IGNORE_DEPRECATED_CPP_DIALECT

#  if THRUST_CPP_DIALECT <= 2003

  struct THRUST_DEPRECATED Cpp03_is_deprecated_in_thrust {};
  Cpp03_is_deprecated_in_thrust x;
  (void)x;

#  elif THRUST_CPP_DIALECT <= 2011

  struct THRUST_DEPRECATED Cpp11_is_deprecated_in_thrust {};
  Cpp11_is_deprecated_in_thrust x;
  (void)x;

#  endif

#endif
}

} // namespace config

THRUST_END_NS
