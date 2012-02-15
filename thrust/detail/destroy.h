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


/*! \file destroy.h
 *  \brief Defines the interface to a function for
 *         dispatching a destructor across a range.
 */

#pragma once

#include <thrust/detail/type_traits.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/detail/dispatch/destroy.h>

namespace thrust
{

namespace detail
{

template<typename ForwardIterator>
  void destroy(ForwardIterator first,
               ForwardIterator last)
{
  typedef typename thrust::iterator_traits<ForwardIterator>::value_type value_type;

  thrust::detail::dispatch::destroy(first, last,
    thrust::detail::has_trivial_destructor<value_type>());
} // end destroy()

} // end detail

} // end thrust

