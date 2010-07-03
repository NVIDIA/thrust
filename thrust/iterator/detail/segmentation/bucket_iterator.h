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

namespace experimental
{

template<typename Derived, typename Base, typename Pointer, typename Value, typename Space, typename Traversal, typename Reference, typename Difference>
class iterator_adaptor;

} // end experimental

namespace detail
{


template<typename Iterator> class segmented_iterator;

template<typename Iterator> struct bucket_iterator {};

template<typename Iterator>
  struct bucket_iterator<thrust::detail::segmented_iterator<Iterator> >
{
  typedef typename thrust::detail::segmented_iterator<Iterator>::bucket_iterator type;
};

template<typename Derived, typename Base, typename Pointer, typename Value, typename Space, typename Traversal, typename Reference, typename Difference>
  struct bucket_iterator<
    thrust::experimental::iterator_adaptor<
      Derived, Base, Pointer, Value, Space, Traversal, Reference, Difference
    >
  >
    : bucket_iterator<Base>
{};

} // end detail

} // end thrust

