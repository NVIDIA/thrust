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
 *  \brief Dispatch layer for destroy.
 */

#pragma once

#include <komrade/detail/type_traits.h>
#include <komrade/for_each.h>

namespace komrade
{

namespace detail
{

namespace dispatch
{

template<typename ForwardIterator>
  void destroy(ForwardIterator first,
               ForwardIterator last,
               komrade::detail::true_type) // has_trivial_destructor
{
  // value_type has a trivial destructor; nothing to do
  ;
} // end destroy()

namespace detail
{

template<typename T>
  struct destroyer
{
  __host__ __device__
  void operator()(T &x) const
  {
    x.~T();
  } // end operator()()
}; // end destroyer

} // end detail

template<typename ForwardIterator>
  void destroy(ForwardIterator first,
               ForwardIterator last,
               komrade::detail::false_type)
{
  typedef typename komrade::iterator_traits<ForwardIterator>::value_type value_type;

  detail::destroyer<value_type> op;
  komrade::for_each(first, last, op);
} // end destroy()

} // end dispatch

} // end detail

} // end komrade

