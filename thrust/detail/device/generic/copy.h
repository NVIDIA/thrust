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


/*! \file copy.h
 *  \brief Generic device implementation of copy.
 */

#pragma once

#include <thrust/detail/device/transform.h>
#include <thrust/functional.h>

namespace thrust
{

namespace detail
{

namespace device
{

// XXX WAR circular #inclusion problems
template<typename InputIterator,
         typename OutputIterator,
         typename UnaryFunction>
  OutputIterator transform(InputIterator,InputIterator,OutputIterator,UnaryFunction);

namespace generic
{

template<typename InputIterator,
         typename OutputIterator>
  OutputIterator copy(InputIterator  first,
                      InputIterator  last,
                      OutputIterator result)
{
  typedef typename thrust::iterator_value<InputIterator>::type T;
  return thrust::detail::device::transform(first, last, result, thrust::identity<T>());
} // end copy()

} // end generic

} // end device

} // end detail

} // end thrust

