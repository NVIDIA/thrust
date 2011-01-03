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

#include <thrust/detail/raw_buffer.h>
#include <thrust/distance.h>
#include <thrust/copy.h>


namespace thrust
{

namespace detail
{


template<typename T, typename Space>
  raw_buffer<T,Space>
    ::raw_buffer(size_type n)
      :super_t(n)
{
  ;
} // end raw_buffer::raw_buffer()


template<typename T, typename Space>
  template<typename InputIterator>
    raw_buffer<T,Space>
      ::raw_buffer(InputIterator first, InputIterator last)
        : super_t()
{
  super_t::allocate(thrust::distance(first,last));
  thrust::copy(first, last, super_t::begin());
} // end raw_buffer::raw_buffer()

} // end detail

} // end thrust

