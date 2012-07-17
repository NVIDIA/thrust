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


/*! \file reverse.inl
 *  \brief Inline file for reverse.h.
 */

#include <thrust/detail/config.h>
#include <thrust/reverse.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/system/detail/generic/select_system.h>
#include <thrust/system/detail/generic/reverse.h>
#include <thrust/system/detail/adl/reverse.h>

namespace thrust
{


template<typename System, typename BidirectionalIterator>
  void reverse(thrust::detail::dispatchable_base<System> &system,
               BidirectionalIterator first,
               BidirectionalIterator last)
{
  using thrust::system::detail::generic::reverse;
  return reverse(system.derived(), first, last);
} // end reverse()


template<typename System, typename BidirectionalIterator, typename OutputIterator>
  OutputIterator reverse_copy(thrust::detail::dispatchable_base<System> &system,
                              BidirectionalIterator first,
                              BidirectionalIterator last,
                              OutputIterator result)
{
  using thrust::system::detail::generic::reverse_copy;
  return reverse_copy(system.derived(), first, last, result);
} // end reverse_copy()


namespace detail
{


template<typename System, typename BidirectionalIterator>
  void strip_const_reverse(const System &system,
                           BidirectionalIterator first,
                           BidirectionalIterator last)
{
  System &non_const_system = const_cast<System&>(system);
  return thrust::reverse(non_const_system, first, last);
} // end strip_const_reverse()


template<typename System, typename BidirectionalIterator, typename OutputIterator>
  OutputIterator strip_const_reverse_copy(const System &system,
                                          BidirectionalIterator first,
                                          BidirectionalIterator last,
                                          OutputIterator result)
{
  System &non_const_system = const_cast<System&>(system);
  return thrust::reverse_copy(non_const_system, first, last, result);
} // end strip_const_reverse_copy()


} // end detail


template<typename BidirectionalIterator>
  void reverse(BidirectionalIterator first,
               BidirectionalIterator last)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<BidirectionalIterator>::type system;

  return thrust::detail::strip_const_reverse(select_system(system()), first, last);
} // end reverse()


template<typename BidirectionalIterator,
         typename OutputIterator>
  OutputIterator reverse_copy(BidirectionalIterator first,
                              BidirectionalIterator last,
                              OutputIterator result)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<BidirectionalIterator>::type system1;
  typedef typename thrust::iterator_system<OutputIterator>::type        system2;

  return thrust::detail::strip_const_reverse_copy(select_system(system1(),system2()), first, last, result);
} // end reverse_copy()


} // end thrust

