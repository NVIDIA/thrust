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

#include <thrust/detail/has_nested_type.h>

namespace thrust
{

namespace experimental
{

template<typename Derived, typename Base, typename Pointer, typename Value, typename Space, typename Traversal, typename Reference, typename Difference>
class iterator_adaptor;

} // end experimental


namespace detail
{


__THRUST_DEFINE_HAS_NESTED_TYPE(has_local_iterator, local_iterator);


template<typename Iterator>
  struct is_segmented
    : has_local_iterator<Iterator>
{};


template<typename Derived, typename Base, typename Pointer, typename Value, typename Space, typename Traversal, typename Reference, typename Difference>
  struct is_segmented<
    thrust::experimental::iterator_adaptor<
      Derived,Base,Pointer,Value,Space,Traversal,Reference,Difference
    >
  >
    : has_local_iterator<Base>
{};


} // end detail

} // end thrust

