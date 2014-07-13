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

#include <thrust/detail/config.h>
#include <thrust/system/cuda/detail/bulk.h>

// we need to carefully undefine and then redefined these macros to ensure that multiple
// versions of cub can coexist in the same program
// push_macro & pop_macro were introduced to gcc in version 4.3

// if the macros are already defined, save them and undefine them

#if !defined(__GNUC__) || (THRUST_GCC_VERSION >= 40300)
#  ifdef CUB_NS_PREFIX
#    pragma push_macro("CUB_NS_PREFIX")
#    undef CUB_NS_PREFIX
#    define CUB_NS_PREFIX_NEEDS_RESTORE
#  endif
#  ifdef CUB_NS_POSTFIX
#    pragma push_macro("CUB_NS_POSTFIX")
#    undef CUB_NS_POSTFIX
#    define CUB_NS_POSTFIX_NEEDS_RESTORE
#  endif
#  ifdef CUB_CDP
#    pragma push_macro("CUB_CDP")
#    undef CUB_CDP
#    define CUB_CDP_NEEDS_RESTORE
#  endif
#  ifdef cub
#    pragma push_macro("cub")
#    undef cub
#    define CUB_NEEDS_RESTORE
#  endif
#endif // __GNUC__

// define the macros while we #include our version of cub
#define CUB_NS_PREFIX namespace thrust { namespace system { namespace cuda { namespace detail {
#define CUB_NS_POSTFIX                  }                  }                }                  }

#if __BULK_HAS_CUDART__
#define CUB_CDP 1
#endif

// rename "cub" so it doesn't collide with another installation elsewhere
#define cub cub_

#include <thrust/system/cuda/detail/cub/util_namespace.cuh>
#include <thrust/system/cuda/detail/cub/cub.cuh>

// undef the top-level namespace name
#undef cub

// undef the macros
#undef CUB_NS_PREFIX
#undef CUB_NS_POSTFIX

#ifdef CUB_CDP
#  undef CUB_CDP
#endif

// redefine the macros if they were defined previously

#if !defined(__GNUC__) || (THRUST_GCC_VERSION >= 40300)
#  ifdef CUB_NS_PREFIX_NEEDS_RESTORE
#    pragma pop_macro("CUB_NS_PREFIX")
#    undef CUB_NS_PREFIX_NEEDS_RESTORE
#  endif
#  ifdef CUB_NS_POSTFIX_NEEDS_RESTORE
#    pragma pop_macro("CUB_NS_POSTFIX")
#    undef CUB_NS_POSTFIX_NEEDS_RESTORE
#  endif
#  ifdef CUB_CDP_NEEDS_RESTORE
#    pragma pop_macro("CUB_CDP")
#    undef CUB_CDP_NEEDS_RESTORE
#  endif
#  ifdef CUB_NEEDS_RESTORE
#    pragma pop_macro("cub")
#    undef CUB_NEEDS_RESTORE
#  endif
#endif // __GNUC__

