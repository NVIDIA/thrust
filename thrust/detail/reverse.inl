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


/*! \file reverse.inl
 *  \brief Inline file for reverse.h.
 */

#include <thrust/detail/config.h>
#include <thrust/reverse.h>
#include <thrust/copy.h>
#include <thrust/iterator/reverse_iterator.h>

namespace thrust
{

template<typename BidirectionalIterator>
  void reverse(BidirectionalIterator first,
               BidirectionalIterator last)
{
  // XXX specialize this implementation later to avoid the temporary buffer
  typedef typename thrust::iterator_traits<BidirectionalIterator>::value_type InputType;
  typedef typename thrust::iterator_space<BidirectionalIterator>::type Space;

  thrust::detail::raw_buffer<InputType,Space> temp(first,last);

  thrust::reverse_copy(temp.begin(), temp.end(), first);
} // end reverse()

template<typename BidirectionalIterator,
         typename OutputIterator>
  OutputIterator reverse_copy(BidirectionalIterator first,
                              BidirectionalIterator last,
                              OutputIterator result)
{
  return thrust::copy(thrust::make_reverse_iterator(last),
                      thrust::make_reverse_iterator(first),
                      result);
} // end reverse_copy()

} // end thrust

