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

#pragma once

namespace thrust
{

template<typename UnaryFunc, typename SegmentedIterator, typename Reference, typename Value> class transform_iterator;


namespace detail
{


template<typename Iterator> class segmented_iterator;

template<typename Iterator> struct local_iterator {};


template<typename Iterator>
  struct local_iterator<thrust::detail::segmented_iterator<Iterator> >
{
  typedef typename thrust::detail::segmented_iterator<Iterator>::local_iterator type;
};


// this metafunction rebinds a segmented transform_iterator to yield a transformed local_iterator
// it essentially removes one layer of segmentation
template<typename UnaryFunc, typename SegmentedIterator, typename Reference, typename Value>
  struct local_iterator<
    thrust::transform_iterator<
      UnaryFunc, SegmentedIterator, Reference, Value
    >
  >
{
  typedef typename thrust::transform_iterator<
    UnaryFunc,
    typename local_iterator<SegmentedIterator>::type,
    Reference,
    Value
  > type;
};


} // end detail

} // end thrust

