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

#include <thrust/detail/config.h>
#include <thrust/iterator/detail/placement/place.h>
#include <thrust/iterator/detail/placement/has_place.h>
#include <thrust/iterator/detail/placement/placed_iterator.h>
#include <thrust/iterator/iterator_adaptor.h>

namespace thrust
{

namespace detail
{

// XXX WAR circular #inclusion with these fowrard declarations
template<typename UnplacedIterator> class placed_iterator;

template<typename Iterator> struct has_place;


template<typename UnplacedIterator>
  typename placed_iterator<UnplacedIterator>::place
    get_place(placed_iterator<UnplacedIterator> iter);


template<typename Derived, typename PlacedBase, typename Pointer, typename Value, typename Space, typename Traversal, typename Reference, typename Difference>
  place_detail::place<Space>
    get_place(thrust::experimental::iterator_adaptor<Derived,PlacedBase,Pointer,Value,Space,Traversal,Reference,Difference> iter,
              typename thrust::detail::enable_if<
                has_place<PlacedBase>
              >::type * = 0);


} // end detail

} // end thrust

#include <thrust/iterator/detail/placement/get_place.inl>

