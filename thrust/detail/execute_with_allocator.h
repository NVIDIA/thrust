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
#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/detail/type_traits/pointer_traits.h>
#include <thrust/detail/allocator/allocator_traits.h>
#include <thrust/detail/util/blocking.h>
#include <thrust/pair.h>

namespace thrust
{
namespace detail
{

template<typename ToPointer, typename FromPointer>
__host__ __device__
ToPointer reinterpret_pointer_cast(FromPointer ptr)
{
  typedef typename thrust::detail::pointer_element<ToPointer>::type to_element;
  return ToPointer(reinterpret_cast<to_element*>(thrust::raw_pointer_cast(ptr)));
}


template<typename Allocator, template <typename> class BaseSystem>
  struct execute_with_allocator
    : BaseSystem<execute_with_allocator<Allocator, BaseSystem> >
{
  Allocator &m_alloc;

  execute_with_allocator(Allocator &alloc)
    : m_alloc(alloc)
  {}

  template<typename T>
    friend thrust::pair<T*,std::ptrdiff_t>
      get_temporary_buffer(execute_with_allocator &system, std::ptrdiff_t n)
  {
    typedef typename thrust::detail::allocator_traits<Allocator> alloc_traits;
    typedef typename alloc_traits::void_pointer                  void_pointer;
    typedef typename alloc_traits::size_type                     size_type;
    typedef typename alloc_traits::value_type                    value_type;

    // how many elements of type value_type do we need to accomodate n elements of type T?
    size_type num_elements = thrust::detail::util::divide_ri(sizeof(T) * n, sizeof(value_type));

    // allocate that many
    void_pointer ptr = alloc_traits::allocate(system.m_alloc, num_elements);

    // return the pointer and the number of elements of type T allocated
    return thrust::make_pair(thrust::detail::reinterpret_pointer_cast<T*>(ptr),n);
  }

  template<typename Pointer>
    friend void return_temporary_buffer(execute_with_allocator &system, Pointer p)
  {
    typedef typename thrust::detail::allocator_traits<Allocator> alloc_traits;
    typedef typename alloc_traits::pointer                       pointer;

    // return the pointer to the allocator
    pointer to_ptr = thrust::detail::reinterpret_pointer_cast<pointer>(p);
    alloc_traits::deallocate(system.m_alloc, to_ptr, 0);
  }
};


} // end detail
} // end thrust

