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

#include <thrust/iterator/iterator_facade.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/range/detail/iterator.h>
#include <thrust/range/detail/value_type.h>
#include <thrust/range/detail/space.h>
#include <thrust/range/detail/reference.h>
#include <thrust/range/detail/difference_type.h>
#include <thrust/range/detail/pointer.h>

namespace thrust
{

namespace detail
{

template<typename Iterator> class segmented_iterator;

template<typename Iterator>
  struct segmented_iterator_base
{
  typedef typename thrust::iterator_value<Iterator>::type                    local_range;
  typedef typename thrust::experimental::range_pointer<local_range>::type    pointer;
  typedef typename thrust::experimental::range_value<local_range>::type      value_type;
  typedef typename thrust::experimental::range_space<local_range>::type      space;
  typedef thrust::bidirectional_traversal_tag                                traversal;
  typedef typename thrust::experimental::range_reference<local_range>::type  reference;
  typedef typename thrust::experimental::range_difference<local_range>::type difference;

  typedef typename thrust::experimental::iterator_facade<
    segmented_iterator<Iterator>,
    pointer,
    value_type,
    space,
    traversal,
    reference,
    difference
  > type;
};

} // end detail

} // end thrust

