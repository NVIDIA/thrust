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
#include <thrust/detail/static_assert.h>
#include <thrust/system/detail/generic/set_operations.h>
#include <thrust/functional.h>

namespace thrust
{
namespace system
{
namespace detail
{
namespace generic
{


template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator>
  OutputIterator set_difference(tag,
                                InputIterator1 first1,
                                InputIterator1 last1,
                                InputIterator2 first2,
                                InputIterator2 last2,
                                OutputIterator result)
{
  typedef typename thrust::iterator_value<InputIterator1>::type value_type;
  return thrust::set_difference(first1, last1, first2, last2, result, thrust::less<value_type>());
} // end set_difference()


template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator>
  OutputIterator set_intersection(tag,
                                  InputIterator1 first1,
                                  InputIterator1 last1,
                                  InputIterator2 first2,
                                  InputIterator2 last2,
                                  OutputIterator result)
{
  typedef typename thrust::iterator_value<InputIterator1>::type value_type;
  return thrust::set_intersection(first1, last1, first2, last2, result, thrust::less<value_type>());
} // end set_intersection()


template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator>
  OutputIterator set_symmetric_difference(tag,
                                          InputIterator1 first1,
                                          InputIterator1 last1,
                                          InputIterator2 first2,
                                          InputIterator2 last2,
                                          OutputIterator result)
{
  typedef typename thrust::iterator_value<InputIterator1>::type value_type;
  return thrust::set_symmetric_difference(first1, last1, first2, last2, result, thrust::less<value_type>());
} // end set_symmetric_difference()


template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator>
  OutputIterator set_union(tag,
                           InputIterator1 first1,
                           InputIterator1 last1,
                           InputIterator2 first2,
                           InputIterator2 last2,
                           OutputIterator result)
{
  typedef typename thrust::iterator_value<InputIterator1>::type value_type;
  return thrust::set_union(first1, last1, first2, last2, result, thrust::less<value_type>());
} // end set_union()


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
  // unimplemented primitive
  THRUST_STATIC_ASSERT( (thrust::detail::depend_on_instantiation<InputIterator1, false>::value) );
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
  // unimplemented primitive
  THRUST_STATIC_ASSERT( (thrust::detail::depend_on_instantiation<InputIterator1, false>::value) );
} // end set_intersection()


template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename StrictWeakOrdering>
  OutputIterator set_symmetric_difference(tag,
                                          InputIterator1 first1,
                                          InputIterator1 last1,
                                          InputIterator2 first2,
                                          InputIterator2 last2,
                                          OutputIterator result,
                                          StrictWeakOrdering comp)
{
  // unimplemented primitive
  THRUST_STATIC_ASSERT( (thrust::detail::depend_on_instantiation<InputIterator1, false>::value) );
} // end set_symmetric_difference()


template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename StrictWeakOrdering>
  OutputIterator set_union(tag,
                           InputIterator1 first1,
                           InputIterator1 last1,
                           InputIterator2 first2,
                           InputIterator2 last2,
                           OutputIterator result,
                           StrictWeakOrdering comp)
{
  // unimplemented primitive
  THRUST_STATIC_ASSERT( (thrust::detail::depend_on_instantiation<InputIterator1, false>::value) );
} // end set_union()


} // end namespace generic
} // end namespace detail
} // end namespace system
} // end namespace thrust

