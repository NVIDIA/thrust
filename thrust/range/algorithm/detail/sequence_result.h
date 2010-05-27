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

#include <thrust/range/detail/is_range.h>
#include <thrust/range/detail/value_type.h>
#include <thrust/range/iterator_range.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/detail/is_space.h>

namespace thrust
{

namespace experimental
{

namespace range
{

namespace detail
{


template<typename ForwardRange>
  struct sequence_result
    : thrust::detail::enable_if<
        thrust::experimental::detail::is_range<ForwardRange>,
        void
      >
{};


template<typename T, typename Space = thrust::any_space_tag>
  struct lazy_sequence_result
    : thrust::detail::enable_if_c<
        !thrust::experimental::detail::is_range<T>::value && thrust::detail::is_space<Space>::value,
        thrust::experimental::iterator_range< thrust::counting_iterator<T,Space> >
      >
{};


} // end detail

} // end range

} // end experimental

} // end thrust

