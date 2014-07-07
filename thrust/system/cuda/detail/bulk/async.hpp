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

#include <thrust/system/cuda/detail/bulk/detail/config.hpp>
#include <thrust/system/cuda/detail/bulk/future.hpp>
#include <thrust/detail/config.h>
#include <thrust/detail/cstdint.h>


BULK_NAMESPACE_PREFIX
namespace bulk
{


template<typename ExecutionGroup, typename Function>
__host__ __device__
future<void> async(ExecutionGroup g, Function f);


template<typename ExecutionGroup, typename Function, typename Arg1>
__host__ __device__
future<void> async(ExecutionGroup g, Function f, Arg1 arg1);


template<typename ExecutionGroup, typename Function, typename Arg1, typename Arg2>
__host__ __device__
future<void> async(ExecutionGroup g, Function f, Arg1 arg1, Arg2 arg2);


template<typename ExecutionGroup, typename Function, typename Arg1, typename Arg2, typename Arg3>
__host__ __device__
future<void> async(ExecutionGroup g, Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3);


template<typename ExecutionGroup, typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4>
__host__ __device__
future<void> async(ExecutionGroup g, Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4);


template<typename ExecutionGroup, typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5>
__host__ __device__
future<void> async(ExecutionGroup g, Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4, Arg5 arg5);


template<typename ExecutionGroup, typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6>
__host__ __device__
future<void> async(ExecutionGroup g, Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4, Arg5 arg5, Arg6 arg6);


template<typename ExecutionGroup, typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6, typename Arg7>
__host__ __device__
future<void> async(ExecutionGroup g, Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4, Arg5 arg5, Arg6 arg6, Arg7 arg7);


template<typename ExecutionGroup, typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6, typename Arg7, typename Arg8>
__host__ __device__
future<void> async(ExecutionGroup g, Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4, Arg5 arg5, Arg6 arg6, Arg7 arg7, Arg8 arg8);


template<typename ExecutionGroup, typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6, typename Arg7, typename Arg8, typename Arg9>
__host__ __device__
future<void> async(ExecutionGroup g, Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4, Arg5 arg5, Arg6 arg6, Arg7 arg7, Arg8 arg8, Arg9 arg9);


template<typename ExecutionGroup, typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6, typename Arg7, typename Arg8, typename Arg9, typename Arg10>
__host__ __device__
future<void> async(ExecutionGroup g, Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4, Arg5 arg5, Arg6 arg6, Arg7 arg7, Arg8 arg8, Arg9 arg9, Arg10 arg10);


} // end bulk
BULK_NAMESPACE_SUFFIX

#include <thrust/system/cuda/detail/bulk/detail/async.inl>

