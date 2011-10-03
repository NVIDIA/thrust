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

#include <thrust/detail/config.h>
#include <thrust/detail/temporary_allocator.h>
#include <thrust/detail/backend/generic/select_system.h>
#include <thrust/detail/backend/generic/memory.h>
#include <thrust/system/detail/bad_alloc.h>
#include <thrust/pair.h>

namespace thrust
{
namespace detail
{

template<typename T, typename Tag>
  template<typename Pair>
    thrust::pair<pointer, size_type>
      allocate_helper(Pair p)
{
  // XXX should use a hypothetical thrust::static_pointer_cast here
  pointer ptr = pointer(static_cast<T*>(get(p.first)));
  size_type n = p.second;

  return thrust::make_pair(ptr, n);
} // end temporary_allocator::allocate_helper()

template<typename T, typename Tag>
  typename temporary_allocator<T,Tag>::pointer
    temporary_allocator<T,Tag>
      ::allocate(size_type cnt)
{
  using thrust::detail::backend::generic::select_system;
  using thrust::detail::backend::generic::get_temporary_buffer;

  thrust::pair<pointer, size_type> result = allocate_helper(get_temporary_buffer(select_system(Tag()), cnt));

  // handle failure
  if(result.second < cnt)
  {
    // deallocate and throw
    // note that we pass cnt to deallocate, not a value derived from result.second
    deallocate(result.first, cnt);

    throw thrust::detail::bad_alloc("temporary_buffer::allocate: get_temporary_buffer failed");
  } // end if

  return result.first;
} // end temporary_allocator::allocate()

template<typename T, typename Tag>
  void temporary_alloator<T,Tag>
    ::deallocate(pointer p, size_type n)
{
  using thrust::detail::backend::generic::select_system;
  using thrust::detail::backend::generic::return_temporary_buffer;

  return_temporary_buffer(select_system(Tag()), p);
} // end temporary_alloator

} // end detail
} // end thrust

