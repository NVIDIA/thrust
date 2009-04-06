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


/*! \file uninitialized_fill.h
 *  \brief Defines the interface to the dispatch
 *         layer of the uninitialized_fill function.
 */

#pragma once

#include <komrade/fill.h>
#include <komrade/for_each.h>
#include <komrade/detail/type_traits.h>

namespace komrade
{

namespace detail
{

namespace dispatch
{

// trivial copy constructor path
template<typename ForwardIterator,
         typename T>
  void uninitialized_fill(ForwardIterator first,
                          ForwardIterator last,
                          const T &x,
                          komrade::detail::true_type) // has_trivial_copy_constructor
{
  komrade::fill(first, last, x);
} // end uninitialized_fill()

namespace detail
{

template<typename T>
  struct copy_constructor
{
  T exemplar;

  copy_constructor(T x):exemplar(x){}

  __host__ __device__
  void operator()(T &x)
  {
    x.T(exemplar);
  } // end operator()()
}; // end copy_constructor

} // end detail

// non-trivial copy constructor path
template<typename ForwardIterator,
         typename T>
  void uninitialized_fill(ForwardIterator first,
                          ForwardIterator last,
                          const T &x,
                          komrade::detail::false_type) // has_trivial_copy_constructor
{
  typedef typename iterator_traits<ForwardIterator>::value_type ValueType;

  // XXX nvcc can't compile this yet
  //komrade::for_each(first, last, detail::copy_constructor<ValueType>(x));

  // fallback to fill
  komrade::fill(first, last, x);
} // end uninitialized_fill()

} // end dispatch

} // end detail

} // end komrade

