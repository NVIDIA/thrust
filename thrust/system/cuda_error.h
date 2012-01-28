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

#include <thrust/detail/config.h>

#if THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_MSVC
#pragma message("-----------------------------------------------------------------------")
#pragma message("| DEPRECATION WARNING:                                                 ")
#pragma message("| thrust/system/cuda_error.h has been deprecated and will be removed   ")
#pragma message("| Use the functionality in thrust/system/cuda/error.h instead          ")
#pragma message("-----------------------------------------------------------------------")
#else
#warning -----------------------------------------------------------------------
#warning | DEPRECATION WARNING: 
#warning | thrust/system/cuda_error.h has been deprecated and will be removed
#warning | Use the functionality in thrust/system/cuda/error.h instead
#warning -----------------------------------------------------------------------
#endif // THRUST_HOST_COMPILER_MSVC

#include <thrust/system/cuda/error.h>

// provide deprecated old names
namespace thrust
{
namespace system
{

namespace cuda_errc = thrust::system::cuda::errc;

// XXX thrust::system::cuda_errc_t is deprecated
//     use thrust::system::cuda::errc_t instead
typedef THRUST_DEPRECATED thrust::system::cuda::errc::errc_t cuda_errc_t;

} // end system

namespace cuda_errc = thrust::system::cuda::errc;

} // end thrust

