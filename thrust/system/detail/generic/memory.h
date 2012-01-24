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


/*! \file generic/memory.h
 *  \brief Generic implementation of memory functions.
 *         Calling some of these is an error. They have no implementation.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/system/detail/generic/tag.h>
#include <thrust/detail/type_traits.h>
#include <thrust/detail/pointer.h>
#include <thrust/pair.h>
#include <thrust/system/detail/generic/type_traits.h>

namespace thrust
{
namespace system
{
namespace detail
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

template<typename T, typename Tag>
  typename thrust::detail::disable_if<
    get_temporary_buffer_exists<
      T, Tag, typename thrust::pointer<T,Tag>::difference_type
    >::value,
    thrust::pair<thrust::pointer<T,Tag>, typename thrust::pointer<T,Tag>::difference_type>
  >::type
    get_temporary_buffer(Tag, typename thrust::pointer<T,Tag>::difference_type n);

template<typename Pointer>
  void return_temporary_buffer(tag, Pointer p);

} // end generic
} // end detail
} // end system
} // end thrust

#include <thrust/system/detail/generic/memory.inl>

