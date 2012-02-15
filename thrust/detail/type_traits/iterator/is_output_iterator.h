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

#include <thrust/detail/config.h>
#include <thrust/detail/type_traits.h>
#include <thrust/detail/type_traits/is_metafunction_defined.h>
#include <thrust/iterator/iterator_traits.h>

namespace thrust
{

namespace detail
{

// XXX this meta function should first check that T is actually an iterator
//
//     if thrust::iterator_value<T> is defined and thrust::iterator_value<T>::type == void
//       return false
//     else
//       return true
template<typename T>
  struct is_output_iterator
    : eval_if<
        is_metafunction_defined<thrust::iterator_value<T> >::value,
        eval_if<
          is_different<
            thrust::iterator_value<T>,
            void
          >::value,
          thrust::detail::false_type,
          thrust::detail::true_type
        >,
        thrust::detail::true_type
      >::type
{
}; // end is_output_iterator

} // end detail

} // end thrust

