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


/*! \file uninitialized_copy.h
 *  \brief Defines the interface to the dispatch
 *         layer of the uninitialized_copy function.
 */

#pragma once

#include <thrust/copy.h>
#include <thrust/for_each.h>
#include <thrust/detail/type_traits.h>

namespace thrust
{

namespace detail
{

namespace dispatch
{

// trivial copy constructor path
template<typename ForwardIterator,
         typename OutputIterator>
  OutputIterator uninitialized_copy(ForwardIterator first,
                                    ForwardIterator last,
                                    OutputIterator result,
                                    thrust::detail::true_type) // has_trivial_copy_constructor
{
  return thrust::copy(first, last, result);
} // end uninitialized_copy()

namespace detail
{

template<typename InputType,
         typename OutputType>
  struct uninitialized_copy_functor
{

  __host__ __device__
  void operator()(const InputType &in, OutputType &out)
  {
    ::new(static_cast<void*>(&out)) OutputType(in);
  } // end operator()()
}; // end uninitialized_copy_functor

} // end detail

// non-trivial copy constructor path
template<typename ForwardIterator,
         typename OutputIterator>
  OutputIterator uninitialized_copy(ForwardIterator first,
                                    ForwardIterator last,
                                    OutputIterator result,
                                    thrust::detail::false_type) // has_trivial_copy_constructor
{
  typedef typename iterator_traits<ForwardIterator>::value_type ValueType;

  // XXX nvcc can't compile this yet
  // XXX we need a binary version of for_each
  //thrust::for_each(first, last, result, detail::uninitialized_copy_functor<ValueType>);

  // fallback to copy
  return thrust::copy(first, last, result);
} // end uninitialized_fill()

} // end dispatch

} // end detail

} // end thrust

