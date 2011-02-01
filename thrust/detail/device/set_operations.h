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

#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/detail/minimum_space.h>
#include <thrust/detail/device/dispatch/set_operations.h>

namespace thrust
{

namespace detail
{

namespace device
{

template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename StrictWeakOrdering>
  OutputIterator set_difference(InputIterator1 first1,
                                InputIterator1 last1,
                                InputIterator2 first2,
                                InputIterator2 last2,
                                OutputIterator result,
                                StrictWeakOrdering comp)
{
  // dispatch on space
  return thrust::detail::device::dispatch::set_difference(first1,last1,first2,last2,result,comp,
    typename thrust::detail::minimum_space<
      typename thrust::iterator_space<InputIterator1>::type,
      typename thrust::iterator_space<InputIterator2>::type,
      typename thrust::iterator_space<OutputIterator>::type
    >::type());
} // end set_difference()

template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename StrictWeakOrdering>
  OutputIterator set_intersection(InputIterator1 first1,
                                  InputIterator1 last1,
                                  InputIterator2 first2,
                                  InputIterator2 last2,
                                  OutputIterator result,
                                  StrictWeakOrdering comp)
{
  // dispatch on space
  return thrust::detail::device::dispatch::set_intersection(first1,last1,first2,last2,result,comp,
    typename thrust::detail::minimum_space<
      typename thrust::iterator_space<InputIterator1>::type,
      typename thrust::iterator_space<InputIterator2>::type,
      typename thrust::iterator_space<OutputIterator>::type
    >::type());
} // end set_intersection()

template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename StrictWeakOrdering>
  OutputIterator set_symmetric_difference(InputIterator1 first1,
                                          InputIterator1 last1,
                                          InputIterator2 first2,
                                          InputIterator2 last2,
                                          OutputIterator result,
                                          StrictWeakOrdering comp)
{
  // dispatch on space
  return thrust::detail::device::dispatch::set_symmetric_difference(first1,last1,first2,last2,result,comp,
    typename thrust::detail::minimum_space<
      typename thrust::iterator_space<InputIterator1>::type,
      typename thrust::iterator_space<InputIterator2>::type,
      typename thrust::iterator_space<OutputIterator>::type
    >::type());
} // end set_symmetric_difference()

template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename StrictWeakOrdering>
  OutputIterator set_union(InputIterator1 first1,
                           InputIterator1 last1,
                           InputIterator2 first2,
                           InputIterator2 last2,
                           OutputIterator result,
                           StrictWeakOrdering comp)
{
  // dispatch on space
  return thrust::detail::device::dispatch::set_union(first1,last1,first2,last2,result,comp,
    typename thrust::detail::minimum_space<
      typename thrust::iterator_space<InputIterator1>::type,
      typename thrust::iterator_space<InputIterator2>::type,
      typename thrust::iterator_space<OutputIterator>::type
    >::type());
} // end set_union()

} // end device

} // end detail

} // end thrust

