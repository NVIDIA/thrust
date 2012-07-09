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


/*! \file fill.h
 *  \brief Device implementation of fill.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/system/cuda/detail/tag.h>

namespace thrust
{
namespace system
{
namespace cuda
{
namespace detail
{

template<typename System, typename ForwardIterator, typename T>
  void fill(dispatchable<System> &system,
            ForwardIterator first,
            ForwardIterator last,
            const T &value);

template<typename System, typename OutputIterator, typename Size, typename T>
  OutputIterator fill_n(dispatchable<System> &system,
                        OutputIterator first,
                        Size n,
                        const T &value);

} // end namespace detail
} // end namespace cuda
} // end namespace system
} // end namespace thrust

#include <thrust/system/cuda/detail/fill.inl>

