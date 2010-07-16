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

#include <thrust/iterator/iterator_adaptor.h>
#include <thrust/iterator/iterator_traits.h>

namespace thrust
{

namespace detail
{

template<typename Iterator> class placed_iterator;

template<typename Iterator>
  struct placed_iterator_base
    : thrust::experimental<
        placed_iterator<Iterator>,
        Iterator,
        typename thrust::iterator_pointer<Iterator>::type,
        typename thrust::iterator_value<Iterator>::type,
        typename thrust::iterator_space<Iterator>::type,
        typename thrust::iterator_traversal<Iterator>::type,
        typename thrust::iterator_reference<Iterator>::type
      >
{
  typedef thrust::experimental<
    placed_iterator<Iterator>,
    Iterator,
    typename thrust::iterator_pointer<Iterator>::type,
    typename thrust::iterator_value<Iterator>::type,
    typename thrust::iterator_space<Iterator>::type,
    typename thrust::iterator_traversal<Iterator>::type,
    typename thrust::iterator_reference<Iterator>::type
  > type;
}; // end placed_iterator_base

} // end detail

} // end thrust

