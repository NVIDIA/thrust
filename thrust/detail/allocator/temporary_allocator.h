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

#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/pointer_base.h>
#include <thrust/detail/type_traits/pointer_traits.h>
#include <thrust/detail/tagged_allocator.h>

namespace thrust
{
namespace detail
{

// XXX the pointer parameter given to tagged_allocator should be related to
//     the type of the expression get_temporary_buffer(Tag(), n).first
//     without decltype, compromise on pointer_base<T,Tag>
template<typename T, typename Tag>
  class temporary_allocator
    : public thrust::detail::tagged_allocator<
               T, Tag, thrust::detail::pointer_base<T,Tag>
             >
{
  private:
    typedef thrust::detail::tagged_allocator<
      T, Tag, thrust::detail::pointer_base<T,Tag>
    > super_t;

  public:
    typename super_t::pointer allocate(typename super_t::size_type cnt);

    void deallocate(typename super_t::pointer p, typename super_t::size_type n);

  private:
    typedef thrust::pair<typename super_t::pointer, typename super_t::size_type> pointer_and_size;

    template<typename Pair>
    static pointer_and_size allocate_helper(Pair p);
};

} // end detail
} // end thrust

#include <thrust/detail/allocator/temporary_allocator.inl>

