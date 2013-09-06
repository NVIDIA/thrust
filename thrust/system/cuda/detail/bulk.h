/*
 *  Copyright 2008-2012 NVIDIA Corporation
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

// we need to carefully undefine and then redefined these macros to ensure that multiple
// versions of bulk can coexist in the same program

// if the macros are already defined, save them and undefine them
#ifdef BULK_NAMESPACE_PREFIX
#  pragma push_macro("BULK_NAMESPACE_PREFIX")
#  undef BULK_NAMESPACE_PREFIX
#  define BULK_NAMESPACE_PREFIX_NEEDS_RESTORE
#endif

#ifdef BULK_NAMESPACE_SUFFIX
#  pragma push_macro("BULK_NAMESPACE_SUFFIX")
#  undef BULK_NAMESPACE_SUFFIX
#  define BULK_NAMESPACE_SUFFIX_NEEDS_RESTORE
#endif

// define the macros while we #include our version of bulk
#define BULK_NAMESPACE_PREFIX namespace thrust { namespace system { namespace cuda { namespace detail {
#define BULK_NAMESPACE_SUFFIX                  }                  }                }                  }

#include <thrust/system/cuda/detail/bulk/bulk.hpp>

// undef the macros
#undef BULK_NAMESPACE_PREFIX
#undef BULK_NAMESPACE_SUFFIX

// redefine the macros if they were defined previously
#ifdef BULK_NAMESPACE_PREFIX_NEEDS_RESTORE
#  pragma pop_macro("BULK_NAMESPACE_PREFIX")
#  undef BULK_NAMESPACE_PREFIX_NEEDS_RESTORE
#endif

#ifdef BULK_NAMESPACE_SUFFIX_NEEDS_RESTORE
#  pragma pop_macro("BULK_NAMESPACE_SUFFIX")
#  undef BULK_NAMESPACE_SUFFIX_NEEDS_RESTORE
#endif

