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


/*! \file device_malloc.inl
 *  \brief Inline file for device_malloc.h.
 */

#include <thrust/detail/config.h>
#include <thrust/device_malloc.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/system/detail/generic/select_system.h>
#include <thrust/system/detail/generic/memory.h>
#include <thrust/detail/malloc_and_free_adl_helper.h>

namespace thrust
{


thrust::device_ptr<void> device_malloc(const std::size_t n)
{
  using thrust::system::detail::generic::select_system;
  using thrust::system::detail::generic::malloc;

  typedef thrust::iterator_system< device_ptr<void> >::type system;

  // XXX should use a hypothetical thrust::static_pointer_cast here
  void* raw_ptr = static_cast<void*>(thrust::raw_pointer_cast(malloc(select_system(system()), n)));

  return thrust::device_ptr<void>(raw_ptr);
} // end device_malloc()


template<typename T>
  thrust::device_ptr<T> device_malloc(const std::size_t n)
{
  thrust::device_ptr<void> void_ptr = thrust::device_malloc(n * sizeof(T));
  return thrust::device_pointer_cast(static_cast<T*>(void_ptr.get()));
} // end device_malloc()


} // end thrust

