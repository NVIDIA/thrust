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
#include <cstdlib> // for malloc & free
#include <thrust/system/cpp/detail/tag.h>

namespace thrust
{
namespace system
{
namespace cpp
{
namespace detail
{


// note that malloc returns a raw pointer to avoid
// depending on the heavyweight thrust/system/cpp/memory.h header
template<typename System>
  void *malloc(dispatchable<System> &, std::size_t n)
{
  return std::malloc(n);
} // end malloc()


template<typename System, typename Pointer>
  void free(dispatchable<System> &, Pointer ptr)
{
  std::free(thrust::raw_pointer_cast(ptr));
} // end free()


} // end detail
} // end cpp
} // end system
} // end thrust

