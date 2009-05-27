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
#include <thrust/iterator/iterator_traits.h>

namespace thrust
{

namespace experimental
{

// forward declaration of counting_iterator
template <typename Incrementable, typename CategoryOrTraversal, typename Difference>
  class counting_iterator;

namespace detail
{

template <typename Incrementable, typename CategoryOrTraversal, typename Difference>
  struct counting_iterator_base
{
  // XXX TODO deduce all this
  //typedef typename detail::ia_dflt_help<
  //    CategoryOrTraversal
  //  , mpl::eval_if<
  //        is_numeric<Incrementable>
  //      , mpl::identity<random_access_traversal_tag>
  //      , iterator_traversal<Incrementable>
  //    >
  //>::type traversal;

  // for the moment, the iterator category is either the default, which is random_access_universal_iterator_tag,
  // or whatever the user provides
  typedef typename detail::ia_dflt_help<
    CategoryOrTraversal,
    identity<thrust::experimental::random_access_universal_iterator_tag>
  >::type traversal;

  // XXX TODO deduce all this
  //typedef typename detail::ia_dflt_help<
  //    Difference
  //  , mpl::eval_if<
  //        is_numeric<Incrementable>
  //      , numeric_difference<Incrementable>
  //      , iterator_difference<Incrementable>
  //    >
  //>::type difference;

  // for the moment, the difference type is either the default, which is ptrdiff_t, or whatever the user provides
  typedef typename detail::ia_dflt_help<
    Difference,
    identity<ptrdiff_t>
  >::type difference;

  typedef iterator_adaptor<
    counting_iterator<Incrementable, CategoryOrTraversal, Difference>, // self
    Incrementable, // Base
    Incrementable, // Value
    traversal,
    Incrementable const &,
    Incrementable *,
    difference
  > type;
}; // end counting_iterator_base

} // end detail

} // end experimental

} // end thrust

