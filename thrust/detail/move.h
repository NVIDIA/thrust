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

#include <thrust/copy.h>
#include <thrust/detail/raw_buffer.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/detail/minimum_space.h>

namespace thrust
{

namespace detail
{

template<typename InputIterator, typename OutputIterator>
  OutputIterator move(InputIterator first, InputIterator last, OutputIterator result)
{
  typedef typename thrust::iterator_space<InputIterator>::type  space1;
  typedef typename thrust::iterator_space<OutputIterator>::type space2;

  typedef typename thrust::detail::minimum_space<space1,space2>::type space;
  typedef typename thrust::iterator_value<InputIterator>::type value_type;

  // do this the brain-dead way for now:
  // make a temporary copy of [first,last), and copy it to first
  // XXX find an in-place parallel algorithm that does this later
  thrust::detail::raw_buffer<value_type, space> temp(first,last);
  return thrust::copy(temp.begin(), temp.end(), result);
} // end move()

} // end detail

} // end thrust

