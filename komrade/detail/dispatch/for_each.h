/*
 *  Copyright 2008-2009 NVIDIA Corporation
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
 *  \brief Defines the interface to the
 *         dispatch layer of the for_each function.
 */

#pragma once

#include <algorithm>
#include <komrade/detail/make_device_dereferenceable.h>
#include <komrade/iterator/iterator_categories.h>
#include <komrade/detail/device/cuda/for_each.h>

namespace komrade
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
void for_each(InputIterator first,
              InputIterator last,
              UnaryFunction f,
              komrade::forward_host_iterator_tag)
{
  std::for_each(first, last, f);
}


/////////////////
// Device Path //
/////////////////
template<typename InputIterator,
         typename UnaryFunction>
void for_each(InputIterator first,
              InputIterator last,
              UnaryFunction f,
              komrade::random_access_device_iterator_tag)
{
  komrade::detail::device::cuda::for_each
      (komrade::detail::make_device_dereferenceable<InputIterator>::transform(first),
       komrade::detail::make_device_dereferenceable<InputIterator>::transform(last),
       f);
}


} // end dispatch

} // end detail

} // end komrade

