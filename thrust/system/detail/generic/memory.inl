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
#include <thrust/system/detail/adl/malloc_and_free.h>
#include <thrust/detail/static_assert.h>
#include <thrust/detail/malloc_and_free.h>

namespace thrust
{
namespace system
{
namespace detail
{
namespace generic
{


template<typename System, typename Size>
  void malloc(thrust::dispatchable<System> &, Size)
{
  // unimplemented
  THRUST_STATIC_ASSERT( (thrust::detail::depend_on_instantiation<Size, false>::value) );
}


template<typename T, typename System>
  thrust::pointer<T,System>
    malloc(thrust::dispatchable<System> &s, std::size_t n)
{
  thrust::pointer<void,System> void_ptr = thrust::malloc(s, sizeof(T) * n);

  return pointer<T,System>(static_cast<T*>(void_ptr.get()));
} // end malloc()


template<typename System, typename Pointer>
  void free(thrust::dispatchable<System> &, Pointer)
{
  // unimplemented
  THRUST_STATIC_ASSERT( (thrust::detail::depend_on_instantiation<Pointer, false>::value) );
}


template<typename System, typename Pointer1, typename Pointer2>
__host__ __device__
void assign_value(thrust::dispatchable<System> &, Pointer1, Pointer2)
{
  // unimplemented
  THRUST_STATIC_ASSERT( (thrust::detail::depend_on_instantiation<Pointer1, false>::value) );
}


template<typename System, typename Pointer>
__host__ __device__
void get_value(thrust::dispatchable<System> &, Pointer)
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


} // end generic
} // end detail
} // end system
} // end thrust

