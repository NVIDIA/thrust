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


/*! \file device_malloc.inl
 *  \brief Inline file for device_malloc.h.
 */

#include <thrust/device_malloc.h>
#include <thrust/detail/device/dispatch/malloc.h>

namespace thrust
{

thrust::device_ptr<void> device_malloc(const std::size_t n)
{
  typedef thrust::iterator_space< device_ptr<void> >::type space;

  return thrust::detail::device::dispatch::malloc<0>(n, space());
} // end device_malloc()

template<typename T>
  thrust::device_ptr<T> device_malloc(const std::size_t n)
{
  return thrust::device_malloc(n * sizeof(T));
} // end device_malloc()

} // end thrust

