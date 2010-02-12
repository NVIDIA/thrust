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


/*! \file distance.h
 *  \brief Dispatch layer to distance function.
 */

#pragma once

#include <iterator>
#include <thrust/detail/device/distance.h>

namespace thrust
{

namespace detail
{

namespace dispatch
{

///////////////    
// Host Path //
///////////////
template<typename InputIterator>
  inline typename thrust::iterator_traits<InputIterator>::difference_type
    distance(InputIterator first, InputIterator last,
             thrust::host_space_tag)
{
  return std::distance(first, last);
} // end distance()


/////////////////
// Device Path //
/////////////////
template<typename InputIterator>
  inline typename thrust::iterator_traits<InputIterator>::difference_type
    distance(InputIterator first, InputIterator last,
             thrust::device_space_tag)
{
  return thrust::detail::device::distance(first, last);
} // end distance()

//////////////
// Any Path //
//////////////
template<typename InputIterator>
  inline typename thrust::iterator_traits<InputIterator>::difference_type
    distance(InputIterator first, InputIterator last,
             thrust::any_space_tag)
{
  // default to device
  return thrust::detail::device::distance(first, last);
} // end distance()

} // end namespace dispatch

} // end namespace detail

} // end namespace thrust

