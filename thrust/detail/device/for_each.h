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


/*! \file for_each.h
 *  \brief Device implementation of for_each.
 */

#pragma once

#include <thrust/detail/device/dispatch/for_each.h>
#include <thrust/iterator/iterator_traits.h>

namespace thrust
{

namespace detail
{

namespace device
{

template<typename InputIterator,
         typename UnaryFunction>
void for_each(InputIterator first,
              InputIterator last,
              UnaryFunction f)
{
  // dispatch on space
  thrust::detail::device::dispatch::for_each(first, last, f,
      typename thrust::iterator_space<InputIterator>::type());
}

} // end namespace device

} // end namespace detail

} // end namespace thrust

