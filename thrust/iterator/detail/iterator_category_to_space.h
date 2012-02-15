/*
 *  Copyright 2008-2012 NVIDIA Corporation
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

#include <thrust/iterator/iterator_categories.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/detail/type_traits.h>
#include <thrust/iterator/detail/device_iterator_category_to_backend_space.h>

namespace thrust
{

// XXX WAR circular #inclusion with forward declarations
struct random_access_universal_iterator_tag;
struct input_universal_iterator_tag;
struct output_universal_iterator_tag;

namespace detail
{

// forward declaration
template <typename> struct is_iterator_space;

template <typename> struct device_iterator_category_to_backend_space;

template<typename Category>
  struct iterator_category_to_space
    // convertible to any iterator?
    : eval_if<
        or_<
          is_convertible<Category, thrust::input_universal_iterator_tag>,
          is_convertible<Category, thrust::output_universal_iterator_tag>
        >::value,

        detail::identity_<thrust::any_space_tag>,

        // convertible to host iterator?
        eval_if<
          or_<
            is_convertible<Category, thrust::input_host_iterator_tag>,
            is_convertible<Category, thrust::output_host_iterator_tag>
          >::value,

          detail::identity_<thrust::host_space_tag>,
          
          // convertible to device iterator?
          eval_if<
            or_<
              is_convertible<Category, thrust::input_device_iterator_tag>,
              is_convertible<Category, thrust::output_device_iterator_tag>
            >::value,

            device_iterator_category_to_backend_space<Category>,

            // unknown space
            void
          > // if device
        > // if host
      > // if any
{
}; // end iterator_category_to_space


template<typename CategoryOrTraversal>
  struct iterator_category_or_traversal_to_space
    : eval_if<
        is_iterator_space<CategoryOrTraversal>::value,
        detail::identity_<CategoryOrTraversal>,
        iterator_category_to_space<CategoryOrTraversal>
      >
{
}; // end iterator_category_or_traversal_to_space

} // end detail

} // end thrust

