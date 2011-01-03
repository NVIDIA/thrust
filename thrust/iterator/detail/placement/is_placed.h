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

#pragma once

#include <thrust/detail/has_nested_type.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/detail/segmentation/segmented_iterator.h>
#include <thrust/iterator/detail/segmentation/local_iterator.h>

namespace thrust
{

namespace detail
{

__THRUST_DEFINE_HAS_NESTED_TYPE(has_place, place);

template<typename Iterator>
  struct has_place<segmented_iterator<Iterator> >
    : has_place<
        typename local_iterator<
          segmented_iterator<Iterator>
        >::type
      >
{};

template<typename UnaryFunc, typename Iterator, typename Reference, typename Value>
  struct has_place<transform_iterator<UnaryFunc,Iterator,Reference,Value> >
    : has_place<
        Iterator
      >::type
{};

} // end detail

} // end thrust

