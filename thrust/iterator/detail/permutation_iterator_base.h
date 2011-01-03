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

#include <thrust/iterator/iterator_adaptor.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/detail/type_traits.h>
#include <thrust/iterator/detail/minimum_space.h>

namespace thrust
{

template<typename,typename> class permutation_iterator;


namespace detail
{

template<typename ElementIterator,
         typename IndexIterator>
  struct permutation_iterator_base
{
  typedef typename thrust::iterator_space<ElementIterator>::type Space1;
  typedef typename thrust::iterator_space<IndexIterator>::type Space2;

  typedef thrust::experimental::iterator_adaptor<
    permutation_iterator<ElementIterator,IndexIterator>,
    IndexIterator,
    typename thrust::iterator_pointer<ElementIterator>::type,
    typename thrust::iterator_value<ElementIterator>::type,
    typename detail::minimum_space<Space1,Space2>::type,
    thrust::use_default,
    typename thrust::iterator_reference<ElementIterator>::type
  > type;
}; // end permutation_iterator_base

} // end detail

} // end thrust

