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
#include <thrust/detail/allocator/malloc_allocator.h>
#include <thrust/system/detail/generic/select_system.h>
#include <thrust/system/detail/generic/memory.h>
#include <thrust/system/detail/bad_alloc.h>
#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/detail/malloc_and_free_adl_helper.h>

namespace thrust
{
namespace detail
{


template<typename T, typename Tag, typename Pointer>
  typename malloc_allocator<T,Tag,Pointer>::pointer
    malloc_allocator<T,Tag,Pointer>
      ::allocate(typename malloc_allocator<T,Tag,Pointer>::size_type cnt)
{
  using thrust::system::detail::generic::select_system;
  using thrust::system::detail::generic::malloc;

  // XXX should use a hypothetical thrust::static_pointer_cast here
  T* result = static_cast<T*>(thrust::raw_pointer_cast(malloc(select_system(Tag()), sizeof(typename super_t::value_type) * cnt)));

  if(result == 0)
  {
    throw thrust::system::detail::bad_alloc("tagged_allocator::allocate: malloc failed");
  } // end if

  return pointer(result);
} // end malloc_allocator::allocate()


template<typename T, typename Tag, typename Pointer>
  void malloc_allocator<T,Tag,Pointer>
    ::deallocate(typename malloc_allocator<T,Tag,Pointer>::pointer p, typename malloc_allocator<T,Tag,Pointer>::size_type n)
{
  using thrust::system::detail::generic::select_system;
  using thrust::system::detail::generic::free;

  free(select_system(Tag()), p);
} // end malloc_allocator


} // end detail
} // end thrust

