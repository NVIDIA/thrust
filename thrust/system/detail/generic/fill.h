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

#include <thrust/detail/internal_functional.h>
#include <thrust/generate.h>
#include <thrust/system/detail/generic/tag.h>

namespace thrust
{
namespace system
{
namespace detail
{
namespace generic
{


template<typename System, typename OutputIterator, typename Size, typename T>
  OutputIterator fill_n(thrust::dispatchable<System> &system,
                        OutputIterator first,
                        Size n,
                        const T &value)
{
  // XXX consider using the placeholder expression _1 = value
  return thrust::generate_n(system, first, n, thrust::detail::fill_functor<T>(value));
}

template<typename System, typename ForwardIterator, typename T>
  void fill(thrust::dispatchable<System> &system,
            ForwardIterator first,
            ForwardIterator last,
            const T &value)
{
  // XXX consider using the placeholder expression _1 = value
  thrust::generate(system, first, last, thrust::detail::fill_functor<T>(value));
}


} // end namespace generic
} // end namespace detail
} // end namespace system
} // end namespace thrust

