/*
 *  Copyright 2008-2009 NVIDIA Corporation
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

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/iterator_adaptor.h>

namespace thrust
{

namespace experimental
{

// forward declaration of constant_iterator
template<typename,typename,typename> class constant_iterator;

namespace detail
{

template<typename Value,
         typename Incrementable,
         typename Space>
  struct constant_iterator_base
{
  typedef Value              value_type;

  // the reference type is just a const reference to the value_type
  typedef const value_type & reference;
  typedef const value_type * pointer;

  // the incrementable type is int unless otherwise specified
  typedef typename ia_dflt_help<
    Incrementable,
    thrust::detail::identity<int>
  >::type incrementable;

  typedef counting_iterator<
    incrementable,
    Space,
    random_access_traversal_tag
  > base_iterator;

  typedef iterator_adaptor<
    constant_iterator<Value, Incrementable, Space>,
    base_iterator,
    pointer,
    value_type,
    typename iterator_space<base_iterator>::type,
    typename iterator_traversal<base_iterator>::type,
    reference
  > type;
}; // end constant_iterator_base

} // end detail
  
} // end experimental

} // end thrust

