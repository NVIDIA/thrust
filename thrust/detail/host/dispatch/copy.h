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

#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/type_traits.h>
#include <thrust/detail/host/detail/general_copy.h>
#include <thrust/detail/host/detail/trivial_copy.h>

namespace thrust
{

namespace detail
{

namespace host
{


namespace dispatch
{


template<typename InputIterator,
         typename OutputIterator>
  OutputIterator copy(InputIterator first,
                      InputIterator last,
                      OutputIterator result,
                      thrust::detail::false_type)
{
  return thrust::detail::host::detail::general_copy(first, last, result);
} // end copy()


template<typename InputIterator,
         typename OutputIterator>
  OutputIterator copy(InputIterator first,
                      InputIterator last,
                      OutputIterator result,
                      thrust::detail::true_type)
{
  return thrust::detail::host::detail::trivial_copy(&*first, &*last, &*result);
} // end copy()



} // end dispatch

} // end host

} // end detail

} // end thrust

