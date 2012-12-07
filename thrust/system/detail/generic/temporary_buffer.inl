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
#include <thrust/system/detail/generic/temporary_buffer.h>
#include <thrust/detail/pointer.h>
#include <thrust/detail/malloc_and_free.h>
#include <thrust/pair.h>

namespace thrust
{
namespace system
{
namespace detail
{
namespace generic
{


template<typename T, typename System>
  thrust::pair<thrust::pointer<T,System>, typename thrust::pointer<T,System>::difference_type>
    get_temporary_buffer(thrust::dispatchable<System> &s, typename thrust::pointer<T,System>::difference_type n)
{
  thrust::pointer<T,System> ptr = thrust::malloc<T>(s, n);

  // check for a failed malloc
  if(!ptr.get())
  {
    n = 0;
  } // end if

  return thrust::make_pair(ptr, n);
} // end get_temporary_buffer()


template<typename System, typename Pointer>
  void return_temporary_buffer(thrust::dispatchable<System> &s, Pointer p)
{
  thrust::free(s, p);
} // end return_temporary_buffer()


} // end generic
} // end detail
} // end system
} // end thrust

