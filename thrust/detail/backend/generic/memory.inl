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
#include <thrust/detail/type_traits/pointer_traits.h>
#include <thrust/detail/backend/generic/memory.h>

namespace thrust
{
namespace detail
{
namespace backend
{
namespace generic
{

namespace detail
{

// define our own raw_pointer_cast to avoid bringing in thrust/device_ptr.h
template<typename Pointer>
  typename thrust::detail::pointer_traits<Pointer>::raw_pointer
    get(Pointer ptr)
{
  return thrust::detail::pointer_traits<Pointer>::get(ptr);
}

} // end detail

template<typename T, typename Tag>
  typename thrust::detail::disable_if<
    get_temporary_buffer_exists<
      T, Tag, typename thrust::detail::pointer_base<T,Tag>::difference_type
    >::value,
    thrust::pair<thrust::detail::pointer_base<T,Tag>, typename thrust::detail::pointer_base<T,Tag>::difference_type>
  >::type
    get_temporary_buffer(Tag, typename thrust::detail::pointer_base<T,Tag>::difference_type n)
{
  typedef thrust::detail::pointer_base<T,Tag> pointer;

  using thrust::detail::backend::generic::select_system;
  using thrust::detail::backend::generic::malloc;

  return thrust::make_pair(pointer(static_cast<T*>(detail::get(malloc(select_system(Tag()), sizeof(T) * n)))), n);
} // end get_temporary_buffer()

template<typename Pointer>
  void return_temporary_buffer(tag, Pointer p)
{
  typedef typename thrust::iterator_space<Pointer>::type Tag;

  using thrust::detail::backend::generic::select_system;
  using thrust::detail::backend::generic::free;

  free(select_system(Tag()), p);
} // end return_temporary_buffer()

} // end generic
} // end backend
} // end detail
} // end thrust

