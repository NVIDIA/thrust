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
#include <thrust/detail/allocator/tagged_allocator.h>
#include <thrust/pair.h>

// XXX WAR circular #inclusion by #including thrust/detail/pointer.h
//     instead of thrust/memory.h
#include <thrust/detail/pointer.h>

namespace thrust
{
namespace detail
{

// XXX the pointer parameter given to tagged_allocator should be related to
//     the type of the expression get_temporary_buffer(Tag(), n).first
//     without decltype, compromise on pointer<T,Tag>
template<typename T, typename Tag>
  class temporary_allocator
    : public thrust::detail::tagged_allocator<
               T, Tag, thrust::pointer<T,Tag>
             >
{
  private:
    typedef thrust::detail::tagged_allocator<
      T, Tag, thrust::pointer<T,Tag>
    > super_t;

  public:
    typedef typename super_t::pointer   pointer;
    typedef typename super_t::size_type size_type;

    pointer allocate(size_type cnt);

    void deallocate(pointer p, size_type n);

  private:
    typedef thrust::pair<pointer, size_type> pointer_and_size;

    template<typename Pair>
    static pointer_and_size allocate_helper(Pair p);
};

} // end detail
} // end thrust

#include <thrust/detail/allocator/temporary_allocator.inl>

