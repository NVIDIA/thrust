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


/*! \file for_each.h
 *  \brief Dispatch layer for the for_each function.
 */

#pragma once

#include <thrust/iterator/iterator_traits.h>

#include <thrust/detail/host/for_each.h>
#include <thrust/detail/device/for_each.h>

namespace thrust
{
namespace detail
{
namespace dispatch
{

///////////////   
// Host Path //
///////////////
template<typename InputIterator,
         typename UnaryFunction>
InputIterator for_each(InputIterator first,
                       InputIterator last,
                       UnaryFunction f,
                       thrust::host_space_tag)
{
  return thrust::detail::host::for_each(first, last, f);
}


template<typename OutputIterator,
         typename Size,
         typename UnaryFunction>
OutputIterator for_each_n(OutputIterator first,
                          Size n,
                          UnaryFunction f,
                          thrust::host_space_tag)
{
  return thrust::detail::host::for_each_n(first, n, f);
}


/////////////////
// Device Path //
/////////////////
template<typename InputIterator,
         typename UnaryFunction>
InputIterator for_each(InputIterator first,
                       InputIterator last,
                       UnaryFunction f,
                       thrust::device_space_tag)
{
  return thrust::detail::device::for_each(first, last, f);
}


template<typename OutputIterator,
         typename Size,
         typename UnaryFunction>
OutputIterator for_each_n(OutputIterator first,
                          Size n,
                          UnaryFunction f,
                          thrust::device_space_tag)
{
  return thrust::detail::device::for_each_n(first, n, f);
}


/////////////////
// Any Path //
/////////////////
template<typename InputIterator,
         typename UnaryFunction>
InputIterator for_each(InputIterator first,
                       InputIterator last,
                       UnaryFunction f,
                       thrust::any_space_tag)
{
  // default to device path
  return thrust::detail::dispatch::for_each(first, last, f, thrust::device_space_tag());
}


template<typename OutputIterator,
         typename Size,
         typename UnaryFunction>
OutputIterator for_each_n(OutputIterator first,
                          Size n,
                          UnaryFunction f,
                          thrust::any_space_tag)
{
  // default to device path
  return thrust::detail::dispatch::for_each_n(first, n, f, thrust::device_space_tag());
}

} // end namespace dispatch
} // end namespace detail
} // end namespace thrust

