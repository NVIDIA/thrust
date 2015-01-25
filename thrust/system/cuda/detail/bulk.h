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

// we need to carefully undefine and then redefined these macros to ensure that multiple
// versions of bulk can coexist in the same program
// push_macro & pop_macro were introduced to gcc in version 4.3

// if the macros are already defined, save them and undefine them

#if !defined(__GNUC__) || (THRUST_GCC_VERSION >= 40300)
#  ifdef BULK_NAMESPACE_PREFIX
#    pragma push_macro("BULK_NAMESPACE_PREFIX")
#    undef BULK_NAMESPACE_PREFIX
#    define BULK_NAMESPACE_PREFIX_NEEDS_RESTORE
#  endif
#  ifdef BULK_NAMESPACE_SUFFIX
#    pragma push_macro("BULK_NAMESPACE_SUFFIX")
#    undef BULK_NAMESPACE_SUFFIX
#    define BULK_NAMESPACE_SUFFIX_NEEDS_RESTORE
#  endif
#endif // __GNUC__

// define the macros while we #include our version of bulk
#define BULK_NAMESPACE_PREFIX namespace thrust { namespace system { namespace cuda { namespace detail {
#define BULK_NAMESPACE_SUFFIX                  }                  }                }                  }

// rename "bulk" so it doesn't collide with another installation elsewhere
#define bulk bulk_

#include <thrust/system/cuda/detail/bulk/bulk.hpp>

// undef the top-level namespace name
#undef bulk

// undef the macros
#undef BULK_NAMESPACE_PREFIX
#undef BULK_NAMESPACE_SUFFIX

// redefine the macros if they were defined previously

#if !defined(__GNUC__) || (THRUST_GCC_VERSION >= 40300)
#  ifdef BULK_NAMESPACE_PREFIX_NEEDS_RESTORE
#    pragma pop_macro("BULK_NAMESPACE_PREFIX")
#    undef BULK_NAMESPACE_PREFIX_NEEDS_RESTORE
#  endif
#  ifdef BULK_NAMESPACE_SUFFIX_NEEDS_RESTORE
#    pragma pop_macro("BULK_NAMESPACE_SUFFIX")
#    undef BULK_NAMESPACE_SUFFIX_NEEDS_RESTORE
#  endif
#endif // __GNUC__

