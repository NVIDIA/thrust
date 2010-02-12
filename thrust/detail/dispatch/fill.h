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


/*! \file fill.h
 *  \brief Defines the interface to the
 *         dispatch layer of the fill function.
 */

#pragma once

#include <thrust/iterator/iterator_categories.h>

#include <algorithm>
#include <thrust/detail/device/fill.h>

namespace thrust
{

namespace detail
{

namespace dispatch
{

///////////////
// Host Path //
///////////////
template<typename ForwardIterator, typename T>
  void fill(ForwardIterator first,
            ForwardIterator last,
            const T &exemplar,
            thrust::host_space_tag)
{
  std::fill(first, last, exemplar);
}

/////////////////
// Device Path //
/////////////////
template<typename ForwardIterator, typename T>
  void fill(ForwardIterator first,
            ForwardIterator last,
            const T &exemplar,
            thrust::device_space_tag)
{
  thrust::detail::device::fill(first, last, exemplar);
}

} // end namespace dispatch

} // end namespace detail

} // end namespace thrust

