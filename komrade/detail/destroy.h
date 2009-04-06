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


/*! \file destroy.h
 *  \brief Defines the interface to a function for
 *         dispatching a destructor across a range.
 */

#pragma once

#include <komrade/detail/type_traits.h>
#include <komrade/iterator/iterator_traits.h>
#include <komrade/detail/dispatch/destroy.h>

namespace komrade
{

namespace detail
{

template<typename ForwardIterator>
  void destroy(ForwardIterator first,
               ForwardIterator last)
{
  typedef typename komrade::iterator_traits<ForwardIterator>::value_type value_type;

  komrade::detail::dispatch::destroy(first, last,
    komrade::detail::has_trivial_destructor<value_type>());
} // end destroy()

} // end detail

} // end komrade

