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


/*! \file distance.h
 *  \brief Dispatch layer to distance function.
 */

#pragma once

#include <iterator>

namespace thrust
{

namespace detail
{

namespace dispatch
{

template<typename InputIterator>
  inline typename thrust::iterator_traits<InputIterator>::difference_type
    distance(InputIterator first, InputIterator last,
             thrust::single_pass_traversal_tag)
{
  typename thrust::iterator_traits<InputIterator>::difference_type result = 0;

  while(first != last)
  {
    ++first;
    ++result;
  }

  return result;
} // end distance()


template<typename InputIterator>
  inline typename thrust::iterator_traits<InputIterator>::difference_type
    distance(InputIterator first, InputIterator last,
             thrust::random_access_traversal_tag)
{
  return last - first;
} // end distance()

} // end namespace dispatch

} // end namespace detail

} // end namespace thrust

