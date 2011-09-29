/*
 *  Copyright 2008-2011 NVIDIA Corporation
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


/*! \file generic/memory.h
 *  \brief Generic implementation of memory functions.
 *         Calling any of these is an error. They have no implementation.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/backend/generic/tag.h>

namespace thrust
{
namespace detail
{
namespace backend
{
namespace generic
{

template<typename Size> void malloc(tag, Size);

template<typename Pointer> void free(tag, Pointer);

template<typename Pointer1, typename Pointer2>
__host__ __device__
void assign_value(tag, Pointer1, Pointer2);

template<typename Pointer>
__host__ __device__
void get_value(tag, Pointer);

template<typename Pointer1, typename Pointer2>
__host__ __device__
void iter_swap(tag, Pointer1, Pointer2);

} // end generic
} // end backend
} // end detail
} // end thrust

