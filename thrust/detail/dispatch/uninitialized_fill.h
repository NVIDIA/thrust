/*
 *  Copyright 2008-2011 NVIDIA Corporation
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

#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/detail/type_traits.h>
#include <thrust/detail/internal_functional.h>

namespace thrust
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
                          thrust::detail::true_type) // has_trivial_copy_constructor
{
  thrust::fill(first, last, x);
} // end uninitialized_fill()

// trivial copy constructor path
template<typename ForwardIterator,
         typename Size,
         typename T>
  ForwardIterator uninitialized_fill_n(ForwardIterator first,
                                       Size n,
                                       const T &x,
                                       thrust::detail::true_type) // has_trivial_copy_constructor
{
  return thrust::fill_n(first, n, x);
} // end uninitialized_fill_n()

// non-trivial copy constructor path
template<typename ForwardIterator,
         typename T>
  void uninitialized_fill(ForwardIterator first,
                          ForwardIterator last,
                          const T &x,
                          thrust::detail::false_type) // has_trivial_copy_constructor
{
  typedef typename iterator_traits<ForwardIterator>::value_type ValueType;

  thrust::for_each(first, last, thrust::detail::uninitialized_fill_functor<ValueType>(x));
} // end uninitialized_fill()

// non-trivial copy constructor path
template<typename ForwardIterator,
         typename Size,
         typename T>
  ForwardIterator uninitialized_fill_n(ForwardIterator first,
                                       Size n,
                                       const T &x,
                                       thrust::detail::false_type) // has_trivial_copy_constructor
{
  typedef typename iterator_traits<ForwardIterator>::value_type ValueType;

  return thrust::detail::for_each_n(first, n, thrust::detail::uninitialized_fill_functor<ValueType>(x));
} // end uninitialized_fill_n()

} // end dispatch

} // end detail

} // end thrust

