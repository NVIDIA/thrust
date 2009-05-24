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


/*! \file distance.h
 *  \brief Dispatch layer to distance function.
 */

#pragma once

#include <thrust/distance.h>
#include <thrust/iterator/iterator_categories.h>
#include <thrust/iterator/iterator_traits.h>
#include <iterator>

namespace thrust
{

namespace detail
{

namespace dispatch
{

// general case
template<typename InputIterator,
         typename IteratorCategory>
  inline typename thrust::iterator_traits<InputIterator>::difference_type
    distance(InputIterator first, InputIterator last,
             IteratorCategory)
{
  return std::distance(first, last);
} // end distance()

// special case: input device iterator
template<typename InputIterator>
  inline typename thrust::iterator_traits<InputIterator>::difference_type
    distance(InputIterator first, InputIterator last,
             thrust::input_device_iterator_tag)
{
  typename thrust::iterator_traits<InputIterator>::difference_type result = 0;

  while(first != last)
  {
    ++first;
    ++result;
  } // end while

  return result;
} // end distance()

// special case: random access device iterator
template<typename InputIterator>
  inline typename thrust::iterator_traits<InputIterator>::difference_type
    distance(InputIterator first, InputIterator last,
             thrust::random_access_device_iterator_tag)
{
  return last - first;
} // end distance()

} // end dispatch

} // end detail

} // end thrust

