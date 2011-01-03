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

#include <thrust/detail/device/for_each.h>
#include <thrust/distance.h>
#include <thrust/detail/device/dispatch/for_each.h>
#include <thrust/iterator/iterator_traits.h>

namespace thrust
{

namespace detail
{

namespace device
{


template<typename OutputIterator,
         typename Size,
         typename UnaryFunction>
OutputIterator for_each_n(OutputIterator first,
                          Size n,
                          UnaryFunction f)
{
  // dispatch on space
  return thrust::detail::device::dispatch::for_each_n(first, n, f,
      typename thrust::iterator_space<OutputIterator>::type());
}


template<typename InputIterator,
         typename UnaryFunction>
InputIterator for_each(InputIterator first,
                       InputIterator last,
                       UnaryFunction f)
{
  // all device iterators are random access right now, so this is safe
  return thrust::detail::device::for_each_n(first, thrust::distance(first,last), f);
}


} // end namespace device

} // end namespace detail

} // end namespace thrust

