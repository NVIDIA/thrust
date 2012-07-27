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


/*! \file distance.inl
 *  \brief Inline file for distance.h
 */

#include <thrust/advance.h>
#include <thrust/system/detail/generic/select_system.h>
#include <thrust/system/detail/generic/distance.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/system/detail/adl/distance.h>

namespace thrust
{


template<typename System, typename InputIterator>
  inline typename thrust::iterator_traits<InputIterator>::difference_type
    distance(thrust::detail::dispatchable_base<System> &system, InputIterator first, InputIterator last)
{
  using thrust::system::detail::generic::distance;
  return distance(system.derived(), first, last);
} // end distance()


namespace detail
{


template<typename System, typename InputIterator>
  inline typename thrust::iterator_traits<InputIterator>::difference_type
    strip_const_distance(const System &system, InputIterator first, InputIterator last)
{
  System &non_const_system = const_cast<System&>(system);
  return thrust::distance(non_const_system, first, last);
} // end distance()


namespace distance_detail
{


// This helper exists purely to avoid warnings concerning null references
// from the null pointer used in distance's dispatch below
template<typename System>
inline __host__ __device__
System &deref(System *ptr)
{
  return *ptr;
} // end deref()


} // end distance_detail


} // end detail


template<typename InputIterator>
  inline typename thrust::iterator_traits<InputIterator>::difference_type
    distance(InputIterator first, InputIterator last)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<InputIterator>::type System;

  // XXX avoid default-constructing a System
  // XXX we justify this hack for distance's dispatch (though, not the dispatch
  // XXX of other algorithms) because distance is more similar to the pointer
  // XXX manipulation functions dispatched by thrust::reference
  // XXX than it is to the algorithms
  System *system = 0;

  return thrust::detail::strip_const_distance(select_system(detail::distance_detail::deref(system)), first, last);
} // end distance()


} // end namespace thrust

