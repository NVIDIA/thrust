/*
 *  Copyright 2008-2012 NVIDIA Corporation
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
#include <thrust/system/detail/generic/tag.h>

namespace thrust
{
namespace system
{
namespace detail
{
namespace generic
{


template<typename System,
         typename InputIterator,
         typename OutputIterator>
  OutputIterator copy(thrust::dispatchable<System> &system,
                      InputIterator  first,
                      InputIterator  last,
                      OutputIterator result);


template<typename System,
         typename InputIterator,
         typename Size,
         typename OutputIterator>
  OutputIterator copy_n(thrust::dispatchable<System> &system,
                        InputIterator  first,
                        Size           n,
                        OutputIterator result);


} // end generic
} // end detail
} // end system
} // end thrust

#include <thrust/system/detail/generic/copy.inl>

