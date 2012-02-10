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
#include <thrust/detail/type_traits/pointer_traits.h>
#include <thrust/system/detail/generic/memory.h>
#include <thrust/system/detail/generic/select_system.h>
#include <thrust/detail/static_assert.h>
#include <thrust/detail/malloc_and_free_adl_helper.h>

namespace thrust
{
namespace system
{
namespace detail
{
namespace generic
{


template<typename Size>
  void malloc(tag, Size)
{
  // unimplemented
  THRUST_STATIC_ASSERT( (thrust::detail::depend_on_instantiation<Size, false>::value) );
}


template<typename Pointer>
  void free(tag, Pointer)
{
  // unimplemented
  THRUST_STATIC_ASSERT( (thrust::detail::depend_on_instantiation<Pointer, false>::value) );
}


template<typename Pointer1, typename Pointer2>
__host__ __device__
void assign_value(tag, Pointer1, Pointer2)
{
  // unimplemented
  THRUST_STATIC_ASSERT( (thrust::detail::depend_on_instantiation<Pointer1, false>::value) );
}


template<typename Pointer>
__host__ __device__
void get_value(tag, Pointer)
{
  // unimplemented
  THRUST_STATIC_ASSERT( (thrust::detail::depend_on_instantiation<Pointer, false>::value) );
}


template<typename Pointer1, typename Pointer2>
__host__ __device__
void iter_swap(tag, Pointer1, Pointer2)
{
  // unimplemented
  THRUST_STATIC_ASSERT( (thrust::detail::depend_on_instantiation<Pointer1, false>::value) );
}


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
      T, Tag, typename thrust::pointer<T,Tag>::difference_type
    >::value,
    thrust::pair<thrust::pointer<T,Tag>, typename thrust::pointer<T,Tag>::difference_type>
  >::type
    get_temporary_buffer(Tag, typename thrust::pointer<T,Tag>::difference_type n)
{
  typedef thrust::pointer<T,Tag> pointer;

  using thrust::system::detail::generic::select_system;
  using thrust::system::detail::generic::malloc;

  return thrust::make_pair(pointer(static_cast<T*>(detail::get(malloc(select_system(Tag()), sizeof(T) * n)))), n);
} // end get_temporary_buffer()


template<typename Pointer>
  void return_temporary_buffer(tag, Pointer p)
{
  typedef typename thrust::iterator_system<Pointer>::type Tag;

  using thrust::system::detail::generic::select_system;
  using thrust::system::detail::generic::free;

  free(select_system(Tag()), p);
} // end return_temporary_buffer()


} // end generic
} // end detail
} // end system
} // end thrust

