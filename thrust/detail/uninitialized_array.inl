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

#include <thrust/detail/uninitialized_array.h>
#include <thrust/distance.h>
#include <thrust/detail/copy.h>


namespace thrust
{

namespace detail
{


template<typename T, typename Space>
  uninitialized_array<T,Space>
    ::uninitialized_array(size_type n)
      :super_t(n)
{
  ;
} // end uninitialized_array::uninitialized_array()


template<typename T, typename Space>
  template<typename InputIterator>
    uninitialized_array<T,Space>
      ::uninitialized_array(InputIterator first, InputIterator last)
        : super_t()
{
  super_t::allocate(thrust::distance(first,last));
  thrust::copy(first, last, super_t::begin());
} // end uninitialized_array::uninitialized_array()

} // end detail

} // end thrust

