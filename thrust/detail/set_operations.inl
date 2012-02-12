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

/*! \file set_operations.inl
 *  \brief Inline file for set_operations.h.
 */

#include <thrust/detail/config.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/system/detail/generic/select_system.h>
#include <thrust/system/detail/generic/set_operations.h>
#include <thrust/detail/adl_helper.h>

namespace thrust
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
  using thrust::system::detail::generic::select_system;
  using thrust::system::detail::generic::set_difference;

  typedef typename thrust::iterator_system<InputIterator1>::type system1;
  typedef typename thrust::iterator_system<InputIterator2>::type system2;
  typedef typename thrust::iterator_system<OutputIterator>::type system3;

  return set_difference(select_system(system1(),system2(),system3()), first1, last1, first2, last2, result, comp);
} // end set_difference()


template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator>
  OutputIterator set_difference(InputIterator1 first1,
                                InputIterator1 last1,
                                InputIterator2 first2,
                                InputIterator2 last2,
                                OutputIterator result)
{
  using thrust::system::detail::generic::select_system;
  using thrust::system::detail::generic::set_difference;

  typedef typename thrust::iterator_system<InputIterator1>::type system1;
  typedef typename thrust::iterator_system<InputIterator2>::type system2;
  typedef typename thrust::iterator_system<OutputIterator>::type system3;

  return set_difference(select_system(system1(),system2(),system3()), first1, last1, first2, last2, result);
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
  using thrust::system::detail::generic::select_system;
  using thrust::system::detail::generic::set_intersection;

  typedef typename thrust::iterator_system<InputIterator1>::type system1;
  typedef typename thrust::iterator_system<InputIterator2>::type system2;
  typedef typename thrust::iterator_system<OutputIterator>::type system3;

  return set_intersection(select_system(system1(),system2(),system3()), first1, last1, first2, last2, result, comp);
} // end set_intersection()


template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator>
  OutputIterator set_intersection(InputIterator1 first1,
                                  InputIterator1 last1,
                                  InputIterator2 first2,
                                  InputIterator2 last2,
                                  OutputIterator result)
{
  using thrust::system::detail::generic::select_system;
  using thrust::system::detail::generic::set_intersection;

  typedef typename thrust::iterator_system<InputIterator1>::type system1;
  typedef typename thrust::iterator_system<InputIterator2>::type system2;
  typedef typename thrust::iterator_system<OutputIterator>::type system3;

  return set_intersection(select_system(system1(),system2(),system3()), first1, last1, first2, last2, result);
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
  using thrust::system::detail::generic::select_system;
  using thrust::system::detail::generic::set_symmetric_difference;

  typedef typename thrust::iterator_system<InputIterator1>::type system1;
  typedef typename thrust::iterator_system<InputIterator2>::type system2;
  typedef typename thrust::iterator_system<OutputIterator>::type system3;

  return set_symmetric_difference(select_system(system1(),system2(),system3()), first1, last1, first2, last2, result, comp);
} // end set_symmetric_difference()


template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator>
  OutputIterator set_symmetric_difference(InputIterator1 first1,
                                          InputIterator1 last1,
                                          InputIterator2 first2,
                                          InputIterator2 last2,
                                          OutputIterator result)
{
  using thrust::system::detail::generic::select_system;
  using thrust::system::detail::generic::set_symmetric_difference;

  typedef typename thrust::iterator_system<InputIterator1>::type system1;
  typedef typename thrust::iterator_system<InputIterator2>::type system2;
  typedef typename thrust::iterator_system<OutputIterator>::type system3;

  return set_symmetric_difference(select_system(system1(),system2(),system3()), first1, last1, first2, last2, result);
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
  using thrust::system::detail::generic::select_system;
  using thrust::system::detail::generic::set_union;

  typedef typename thrust::iterator_system<InputIterator1>::type system1;
  typedef typename thrust::iterator_system<InputIterator2>::type system2;
  typedef typename thrust::iterator_system<OutputIterator>::type system3;

  return set_union(select_system(system1(),system2(),system3()), first1, last1, first2, last2, result, comp);
} // end set_union()


template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator>
  OutputIterator set_union(InputIterator1 first1,
                           InputIterator1 last1,
                           InputIterator2 first2,
                           InputIterator2 last2,
                           OutputIterator result)
{
  using thrust::system::detail::generic::select_system;
  using thrust::system::detail::generic::set_union;

  typedef typename thrust::iterator_system<InputIterator1>::type system1;
  typedef typename thrust::iterator_system<InputIterator2>::type system2;
  typedef typename thrust::iterator_system<OutputIterator>::type system3;

  return set_union(select_system(system1(),system2(),system3()), first1, last1, first2, last2, result);
} // end set_union()


} // end thrust

