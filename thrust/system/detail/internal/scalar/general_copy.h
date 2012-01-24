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

/*! \file general_copy.h
 *  \brief Sequential copy algorithms for general iterators.
 */

#pragma once

#include <thrust/detail/config.h>

namespace thrust
{
namespace system
{
namespace detail
{
namespace internal
{
namespace scalar
{

template<typename InputIterator,
         typename OutputIterator>
  OutputIterator general_copy(InputIterator first,
                              InputIterator last,
                              OutputIterator result)
{
  for(; first != last; ++first, ++result)
    *result = *first;
  return result;
} // end general_copy()


template<typename InputIterator,
         typename Size,
         typename OutputIterator>
  OutputIterator general_copy_n(InputIterator first,
                                Size n,
                                OutputIterator result)
{
  for(; n > Size(0); ++first, ++result, --n)
    *result = *first;
  return result;
} // end general_copy_n()

} // end namespace scalar
} // end namespace internal
} // end namespace detail
} // end namespace system
} // end namespace thrust

