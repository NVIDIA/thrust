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

#include <thrust/detail/config.h>
#include <thrust/detail/type_traits.h>
#include <thrust/iterator/detail/host_space_tag.h>
#include <thrust/iterator/detail/device_space_tag.h>
#include <thrust/iterator/detail/any_space_tag.h>
#include <thrust/iterator/iterator_categories.h>
#include <thrust/iterator/detail/iterator_traversal_tags.h>
#include <thrust/iterator/detail/is_iterator_category.h>
#include <thrust/iterator/detail/iterator_category_to_traversal.h>

namespace thrust
{

namespace detail
{

template<typename Category, typename Space, typename Traversal>
  struct iterator_category_with_space_and_traversal
    : Category
{
}; // end iterator_category_with_space_and_traversal

// specialize iterator_category_to_space for iterator_category_with_space_and_traversal
template<typename Category> struct iterator_category_to_space;

template<typename Category, typename Space, typename Traversal>
  struct iterator_category_to_space<iterator_category_with_space_and_traversal<Category,Space,Traversal> >
{
  typedef Space type;
}; // end iterator_category_with_space_and_traversal

} // end detail

namespace experimental
{

namespace detail
{

// adapted from http://www.boost.org/doc/libs/1_37_0/libs/iterator/doc/iterator_facade.html#iterator-category
//
// in our implementation, R need not be a reference type to result in a category
// derived from forward_XXX_iterator_tag
//
// iterator-category(T,V,R) :=
//   if(T is convertible to input_host_iterator_tag
//      || T is convertible to output_host_iterator_tag
//      || T is convertible to input_device_iterator_tag
//      || T is convertible to output_device_iterator_tag
//   )
//     return T
//
//   else if (T is not convertible to incrementable_traversal_tag)
//     the program is ill-formed
//
//   else return a type X satisfying the following two constraints:
//
//     1. X is convertible to X1, and not to any more-derived
//        type, where X1 is defined by:
//
//        if (T is convertible to forward_traversal_tag)
//        {
//          if (T is convertible to random_access_traversal_tag)
//            X1 = random_access_host_iterator_tag
//          else if (T is convertible to bidirectional_traversal_tag)
//            X1 = bidirectional_host_iterator_tag
//          else
//            X1 = forward_host_iterator_tag
//        }
//        else
//        {
//          if (T is convertible to single_pass_traversal_tag
//              && R is convertible to V)
//            X1 = input_host_iterator_tag
//          else
//            X1 = T
//        }
//
//     2. category-to-traversal(X) is convertible to the most
//        derived traversal tag type to which X is also convertible,
//        and not to any more-derived traversal tag type.


template<typename Space, typename Traversal, typename ValueParam, typename Reference>
  struct iterator_facade_default_category;

//
// Convert an iterator_facade's traversal category, Value parameter,
// and ::reference type to an appropriate old-style category.
//
// If writability has been disabled per the above metafunction, the
// result will not be convertible to output_iterator_tag.
//
// Otherwise, if Traversal == single_pass_traversal_tag, the following
// conditions will result in a tag that is convertible both to
// input_iterator_tag and output_iterator_tag:
//
//    1. Reference is a reference to non-const
//    2. Reference is not a reference and is convertible to Value
//


// this is the function for standard space iterators
template<typename Traversal, typename ValueParam, typename Reference>
  struct iterator_facade_default_category_std :
    thrust::detail::eval_if<
      thrust::detail::is_convertible<Traversal, thrust::forward_traversal_tag>::value,
      thrust::detail::eval_if<
        thrust::detail::is_convertible<Traversal, thrust::random_access_traversal_tag>::value,
        thrust::detail::identity_<std::random_access_iterator_tag>,
        thrust::detail::eval_if<
          thrust::detail::is_convertible<Traversal, thrust::bidirectional_traversal_tag>::value,
          thrust::detail::identity_<std::bidirectional_iterator_tag>,
          thrust::detail::identity_<std::forward_iterator_tag>
        >
      >,
      thrust::detail::eval_if<
        thrust::detail::and_<
          thrust::detail::is_convertible<Traversal, thrust::single_pass_traversal_tag>,
          thrust::detail::is_convertible<Reference, ValueParam>
        >::value,
        thrust::detail::identity_<std::input_iterator_tag>,
        thrust::detail::identity_<Traversal>
      >
    >
{
}; // end iterator_facade_default_category_std


// this is the function for host space iterators
template<typename Traversal, typename ValueParam, typename Reference>
  struct iterator_facade_default_category_host :
    thrust::detail::eval_if<
      thrust::detail::is_convertible<Traversal, thrust::forward_traversal_tag>::value,
      thrust::detail::eval_if<
        thrust::detail::is_convertible<Traversal, thrust::random_access_traversal_tag>::value,
        thrust::detail::identity_<thrust::random_access_host_iterator_tag>,
        thrust::detail::eval_if<
          thrust::detail::is_convertible<Traversal, thrust::bidirectional_traversal_tag>::value,
          thrust::detail::identity_<thrust::bidirectional_host_iterator_tag>,
          thrust::detail::identity_<thrust::forward_host_iterator_tag>
        >
      >,
      thrust::detail::eval_if<
        thrust::detail::and_<
          thrust::detail::is_convertible<Traversal, thrust::single_pass_traversal_tag>,
          thrust::detail::is_convertible<Reference, ValueParam>
        >::value,
        thrust::detail::identity_<thrust::input_host_iterator_tag>,
        thrust::detail::identity_<Traversal>
      >
    >
{
}; // end iterator_facade_default_category_host


// this is the function for device space iterators
template<typename Traversal, typename ValueParam, typename Reference>
  struct iterator_facade_default_category_device :
    thrust::detail::eval_if<
      thrust::detail::is_convertible<Traversal, thrust::forward_traversal_tag>::value,
      thrust::detail::eval_if<
        thrust::detail::is_convertible<Traversal, thrust::random_access_traversal_tag>::value,
        thrust::detail::identity_<thrust::random_access_device_iterator_tag>,
        thrust::detail::eval_if<
          thrust::detail::is_convertible<Traversal, thrust::bidirectional_traversal_tag>::value,
          thrust::detail::identity_<thrust::bidirectional_device_iterator_tag>,
          thrust::detail::identity_<thrust::forward_device_iterator_tag>
        >
      >,
      thrust::detail::eval_if<
        thrust::detail::and_<
          thrust::detail::is_convertible<Traversal, thrust::single_pass_traversal_tag>,
          thrust::detail::is_convertible<Reference, ValueParam>
        >::value,
        thrust::detail::identity_<thrust::input_device_iterator_tag>,
        thrust::detail::identity_<Traversal>
      >
    >
{
}; // end iterator_facade_default_category_device


// this is the function for any space iterators
template<typename Traversal, typename ValueParam, typename Reference>
  struct iterator_facade_default_category_any :
    thrust::detail::eval_if<

      thrust::detail::is_convertible<Traversal, thrust::forward_traversal_tag>::value,

      thrust::detail::eval_if<
        thrust::detail::is_convertible<Traversal, thrust::random_access_traversal_tag>::value,
        thrust::detail::identity_<thrust::random_access_universal_iterator_tag>,

        thrust::detail::eval_if<
          thrust::detail::is_convertible<Traversal, thrust::bidirectional_traversal_tag>::value,
          thrust::detail::identity_<thrust::bidirectional_universal_iterator_tag>,
          thrust::detail::identity_<thrust::forward_universal_iterator_tag>
        >
      >,

      thrust::detail::eval_if<
        thrust::detail::and_<
          thrust::detail::is_convertible<Traversal, thrust::single_pass_traversal_tag>,
          thrust::detail::is_convertible<Reference, ValueParam>
        >::value,
        thrust::detail::identity_<thrust::input_universal_iterator_tag>,
        thrust::detail::identity_<Traversal>
      >
    >
{
}; // end iterator_facade_default_category_any


template<typename Space, typename Traversal, typename ValueParam, typename Reference>
  struct iterator_facade_default_category
      // check for any space
    : thrust::detail::eval_if<
        thrust::detail::is_convertible<Space, thrust::any_space_tag>::value,
        iterator_facade_default_category_any<Traversal, ValueParam, Reference>,

        // check for host space
        thrust::detail::eval_if<
          thrust::detail::is_convertible<Space, thrust::host_space_tag>::value,
          iterator_facade_default_category_host<Traversal, ValueParam, Reference>,

          // check for device space
          thrust::detail::eval_if<
            thrust::detail::is_convertible<Space, thrust::device_space_tag>::value,
            iterator_facade_default_category_device<Traversal, ValueParam, Reference>,

            // if we don't recognize the space, get a standard iterator category
            // and combine it with Space & Traversal
            thrust::detail::identity_<
              thrust::detail::iterator_category_with_space_and_traversal<
                typename iterator_facade_default_category_std<Traversal, ValueParam, Reference>::type,
                Space,
                Traversal
              >
            >
          >
        >
      >
{};


template<typename Space, typename Traversal, typename ValueParam, typename Reference>
  struct iterator_facade_category_impl
{
  typedef typename iterator_facade_default_category<
    Space,Traversal,ValueParam,Reference
  >::type category;

  // we must be able to deduce both Traversal & Space from category
  // otherwise, munge them all together
  typedef typename thrust::detail::eval_if<
    thrust::detail::and_<
      thrust::detail::is_same<
        Traversal,
        typename thrust::detail::iterator_category_to_traversal<category>::type
      >,
      thrust::detail::is_same<
        Space,
        typename thrust::detail::iterator_category_to_space<category>::type
      >
    >::value,
    thrust::detail::identity_<category>,
    thrust::detail::identity_<thrust::detail::iterator_category_with_space_and_traversal<category,Space,Traversal> >
  >::type type;
}; // end iterator_facade_category_impl


template<typename CategoryOrSpace,
         typename CategoryOrTraversal,
         typename ValueParam,
         typename Reference>
  struct iterator_facade_category
{
  typedef typename
  thrust::detail::eval_if<
    thrust::detail::is_iterator_category<CategoryOrTraversal>::value,
    thrust::detail::identity_<CategoryOrTraversal>, // categories are fine as-is
    iterator_facade_category_impl<CategoryOrSpace, CategoryOrTraversal, ValueParam, Reference>
  >::type type;
}; // end iterator_facade_category

} // end detail

} // end experimental

} // end thrust

