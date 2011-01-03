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
#include <cstring>


namespace thrust
{

namespace detail
{

namespace host
{

namespace detail
{


template<typename T>
  T *trivial_copy_n(const T *first,
                    std::ptrdiff_t n,
                    T *result)
{
  std::memmove(result, first, n * sizeof(T));
  return result + n;
} // end trivial_copy_n()


} // end detail

} // end host

} // end detail

} // end thrust


