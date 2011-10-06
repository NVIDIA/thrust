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

#include <thrust/detail/backend/dereference.h>

namespace thrust
{
namespace detail
{
namespace backend
{
namespace cpp
{


namespace detail
{

template<typename OutputIterator,
         typename Size,
         typename UnaryFunction>
OutputIterator for_each_n(OutputIterator first,
                          Size n,
                          UnaryFunction f)
{
  for(Size i = 0; i != n; i++)
  {
    // we can dereference an OutputIterator if f does not
    // try to use the reference for anything besides assignment
    f(backend::dereference(first));
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
  for(; first != last; ++first)
  {
    f(backend::dereference(first));
  }

  return first;
} // end for_each()

} // end namespace detail

template<typename InputIterator,
         typename UnaryFunction>
void for_each(tag,
              InputIterator first,
              InputIterator last,
              UnaryFunction f)
{
  thrust::detail::backend::cpp::detail::for_each(first, last, f);
} // end for_each()

} // end namespace cpp
} // end namespace backend
} // end namespace detail
} // end namespace thrust

