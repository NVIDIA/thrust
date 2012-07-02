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

#include <thrust/detail/config.h>
#include <thrust/detail/allocator/temporary_allocator.h>
#include <thrust/system/detail/generic/select_system.h>
#include <thrust/system/detail/generic/memory.h>
#include <thrust/system/detail/bad_alloc.h>
#include <thrust/pair.h>
#include <thrust/detail/raw_pointer_cast.h>

namespace thrust
{
namespace detail
{
namespace temporary_allocator_detail
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


// XXX eliminate this should we ever add thrust::get_temporary_buffer
template<typename T, typename System>
  thrust::pair<thrust::pointer<T,System>, typename thrust::pointer<T,System>::difference_type>
    strip_const_get_temporary_buffer(const System &system, typename thrust::pointer<T,System>::difference_type n)
{
  System &non_const_system = const_cast<System&>(system);

  using thrust::system::detail::generic::get_temporary_buffer;

  return down_cast_pair<T,System>(get_temporary_buffer<T>(non_const_system,n));
} // end strip_const_get_temporary_buffer()


// XXX eliminate this should we ever add thrust::return_temporary_buffer
template<typename System, typename Pointer>
void strip_const_return_temporary_buffer(const System &system, Pointer p)
{
  System &non_const_system = const_cast<System&>(system);

  using thrust::system::detail::generic::return_temporary_buffer;

  return_temporary_buffer(non_const_system, p);
} // end strip_const_return_temporary_buffer()


} // end temporary_allocator_detail


template<typename T, typename Tag>
  typename temporary_allocator<T,Tag>::pointer
    temporary_allocator<T,Tag>
      ::allocate(typename temporary_allocator<T,Tag>::size_type cnt)
{
  using thrust::system::detail::generic::select_system;

  pointer_and_size result = temporary_allocator_detail::strip_const_get_temporary_buffer<T>(select_system(Tag()), cnt);

  // handle failure
  if(result.second < cnt)
  {
    // deallocate and throw
    // note that we pass cnt to deallocate, not a value derived from result.second
    deallocate(result.first, cnt);

    throw thrust::system::detail::bad_alloc("temporary_buffer::allocate: get_temporary_buffer failed");
  } // end if

  return result.first;
} // end temporary_allocator::allocate()

template<typename T, typename Tag>
  void temporary_allocator<T,Tag>
    ::deallocate(typename temporary_allocator<T,Tag>::pointer p, typename temporary_allocator<T,Tag>::size_type n)
{
  using thrust::system::detail::generic::select_system;

  temporary_allocator_detail::strip_const_return_temporary_buffer(select_system(Tag()), p);
} // end temporary_allocator

} // end detail
} // end thrust

