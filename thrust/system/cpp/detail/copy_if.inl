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

#include <thrust/detail/config.h>
#include <thrust/system/cpp/detail/copy_if.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/detail/wrapped_function.h>

namespace thrust
{
namespace system
{
namespace cpp
{
namespace detail
{


template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename Predicate>
  OutputIterator copy_if(tag,
                         InputIterator1 first,
                         InputIterator1 last,
                         InputIterator2 stencil,
                         OutputIterator result,
                         Predicate pred)
{
  // wrap pred
  thrust::detail::host_wrapped_unary_function<
    Predicate,
    typename thrust::iterator_reference<InputIterator2>::type, 
    bool
  > wrapped_pred(pred);

  while(first != last)
  {
    if(wrapped_pred(*stencil))
    {
      *result = *first;
      ++result;
    } // end if

    ++first;
    ++stencil;
  } // end while

  return result;
} // end copy_if()


} // end detail
} // end cpp
} // end system
} // end thrust

