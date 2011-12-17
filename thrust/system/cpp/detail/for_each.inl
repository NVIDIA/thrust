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

#pragma once

#include <thrust/detail/wrapped_function.h>

namespace thrust
{
namespace system
{
namespace cpp
{
namespace detail
{

template<typename InputIterator,
         typename Size,
         typename UnaryFunction>
InputIterator for_each_n(tag,
                         InputIterator first,
                         Size n,
                         UnaryFunction f)
{
  // create a wrapped function for f
  typedef typename thrust::iterator_reference<InputIterator>::type reference;
  thrust::detail::host_wrapped_unary_function<UnaryFunction,reference,void> wrapped_f(f);

  for(Size i = 0; i != n; i++)
  {
    // we can dereference an OutputIterator if f does not
    // try to use the reference for anything besides assignment
    wrapped_f(*first);
    ++first;
  }

  return first;
} // end for_each_n()

template<typename InputIterator,
         typename UnaryFunction>
InputIterator for_each(tag,
                       InputIterator first,
                       InputIterator last,
                       UnaryFunction f)
{
  // create a wrapped function for f
  typedef typename thrust::iterator_reference<InputIterator>::type reference;
  thrust::detail::host_wrapped_unary_function<UnaryFunction,reference,void> wrapped_f(f);

  for(; first != last; ++first)
  {
    wrapped_f(*first);
  }

  return first;
} // end for_each()

} // end namespace detail
} // end namespace cpp
} // end namespace system
} // end namespace thrust

