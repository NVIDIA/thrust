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


/*! \file fill.inl
 *  \brief Inline file for fill.h.
 */

#include <thrust/fill.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/detail/dispatch/fill.h>

namespace thrust
{

template<typename ForwardIterator, typename T>
  void fill(ForwardIterator first,
            ForwardIterator last,
            const T &exemplar)
{
  // dispatch on space
  thrust::detail::dispatch::fill(first, last, exemplar,
    typename thrust::iterator_space<ForwardIterator>::type());
} // end fill()

template<typename OutputIterator, typename Size, typename T>
  OutputIterator fill_n(OutputIterator first,
                        Size n,
                        const T &exemplar)
{
  // dispatch on space
  return thrust::detail::dispatch::fill_n(first, n, exemplar,
    typename thrust::iterator_space<OutputIterator>::type());
} // end fill()

} // end namespace thrust

