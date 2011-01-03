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

/*! \file copy.h
 *  \brief Host implementations of copy functions.
 */

#pragma once

#include <thrust/detail/host/dispatch/copy.h>
#include <thrust/detail/dispatch/is_trivial_copy.h>

namespace thrust
{

namespace detail
{

namespace host
{


template<typename InputIterator,
         typename OutputIterator>
  OutputIterator copy(InputIterator first,
                      InputIterator last,
                      OutputIterator result)
{
  return thrust::detail::host::dispatch::copy(first, last, result,
      typename thrust::detail::dispatch::is_trivial_copy<InputIterator,OutputIterator>::type());
} // end copy()

template<typename InputIterator,
         typename Size,
         typename OutputIterator>
  OutputIterator copy_n(InputIterator first,
                        Size n,
                        OutputIterator result)
{
  return thrust::detail::host::dispatch::copy_n(first, n, result,
      typename thrust::detail::dispatch::is_trivial_copy<InputIterator,OutputIterator>::type());
} // end copy()


template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename Predicate>
  OutputIterator copy_if(InputIterator1 first,
                         InputIterator1 last,
                         InputIterator2 stencil,
                         OutputIterator result,
                         Predicate pred)
{
  while(first != last)
  {
    if(pred(*stencil))
    {
      *result = *first;
      ++result;
    } // end if

    ++first;
    ++stencil;
  } // end while

  return result;
} // end copy_if()

} // end host

} // end detail

} // end thrust

