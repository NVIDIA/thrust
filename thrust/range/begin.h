/*
 *  Copyright 2008-2010 NVIDIA Corporation
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

//  Copyright Thorsten Ottosen 2003-2004. Use, modification and
//  distribution is subject to the Boost Software License, Version
//  1.0. (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
// For more information, see http://www.boost.org/libs/range/

#pragma once

#include <thrust/detail/config.h>
#include <thrust/range/detail/iterator.h>

#include <cstddef> // for std::size_t


namespace thrust
{

namespace experimental
{


// default
template<typename Range>
__host__ __device__
  inline typename range_iterator<Range>::type
    begin(Range &r)
{
// XXX WAR nvcc's issues with calling __host__ or __device__ from __host__ __device__
#ifndef __CUDA_ARCH__
  return r.begin();
#else
  return typename range_iterator<Range>::type();
#endif
}


template<typename Range>
__host__ __device__
  inline typename range_iterator<const Range>::type
    begin(const Range &r)
{
// XXX WAR nvcc's issues with calling __host__ or __device__ from __host__ __device__
#ifndef __CUDA_ARCH__
  return r.begin();
#else
  return typename range_iterator<Range>::type();
#endif
}


// arrays
template<typename T, std::size_t sz>
__host__ __device__
  inline T* begin(T (&a)[sz])
{
  return a;
}


// const arrays
// XXX possibly discardable per note in Boost
template<typename T, std::size_t sz>
__host__ __device__
  inline const T* begin(const T (&a)[sz])
{
  return a;
}


} // end experimental

} // end thrust

