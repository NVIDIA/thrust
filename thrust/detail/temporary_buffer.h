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
#include <thrust/detail/dispatchable.h>
#include <thrust/pair.h>
#include <thrust/detail/pointer.h>
#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/system/detail/generic/temporary_buffer.h>
#include <thrust/system/detail/adl/temporary_buffer.h>

namespace thrust
{
namespace detail
{
namespace get_temporary_buffer_detail
{


template<typename T, typename System, typename Pair>
  thrust::pair<thrust::pointer<T,System>, typename thrust::pointer<T,System>::difference_type>
    down_cast_pair(Pair p)
{
  // XXX should use a hypothetical thrust::static_pointer_cast here
  thrust::pointer<T,System> ptr = thrust::pointer<T,System>(static_cast<T*>(thrust::raw_pointer_cast(p.first)));

  typedef thrust::pair<thrust::pointer<T,System>, typename thrust::pointer<T,System>::difference_type> result_type;
  return result_type(ptr, p.second);
} // end down_cast_pair()


} // end get_temporary_buffer_detail
} // end detail


template<typename T, typename System>
  thrust::pair<thrust::pointer<T,System>, typename thrust::pointer<T,System>::difference_type>
    get_temporary_buffer(const thrust::detail::dispatchable_base<System> &system, typename thrust::pointer<T,System>::difference_type n)
{
  using thrust::system::detail::generic::get_temporary_buffer;

  return thrust::detail::get_temporary_buffer_detail::down_cast_pair<T,System>(get_temporary_buffer<T>(thrust::detail::derived_cast(thrust::detail::strip_const(system)), n));
} // end get_temporary_buffer()


template<typename System, typename Pointer>
  void return_temporary_buffer(const thrust::detail::dispatchable_base<System> &system, Pointer p)
{
  using thrust::system::detail::generic::return_temporary_buffer;

  return return_temporary_buffer(thrust::detail::derived_cast(thrust::detail::strip_const(system)), p);
} // end return_temporary_buffer()


} // end thrust

