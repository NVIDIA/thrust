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


/*! \file uninitialized_fill.inl
 *  \brief Inline file for uninitialized_fill.h.
 */

#include <thrust/uninitialized_fill.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/system/detail/generic/select_system.h>
#include <thrust/system/detail/generic/uninitialized_fill.h>
#include <thrust/system/detail/adl/uninitialized_fill.h>

namespace thrust
{


template<typename System, typename ForwardIterator, typename T>
  void uninitialized_fill(thrust::detail::dispatchable_base<System> &system,
                          ForwardIterator first,
                          ForwardIterator last,
                          const T &x)
{
  using thrust::system::detail::generic::uninitialized_fill;
  return uninitialized_fill(system.derived(), first, last, x);
} // end uninitialized_fill()


template<typename System, typename ForwardIterator, typename Size, typename T>
  ForwardIterator uninitialized_fill_n(thrust::detail::dispatchable_base<System> &system,
                                       ForwardIterator first,
                                       Size n,
                                       const T &x)
{
  using thrust::system::detail::generic::uninitialized_fill_n;
  return uninitialized_fill_n(system.derived(), first, n, x);
} // end uninitialized_fill_n()


namespace detail
{


template<typename System, typename ForwardIterator, typename T>
  void strip_const_uninitialized_fill(const System &system,
                                      ForwardIterator first,
                                      ForwardIterator last,
                                      const T &x)
{
  System &non_const_system = const_cast<System&>(system);
  return thrust::uninitialized_fill(non_const_system, first, last, x);
} // end strip_const_uninitialized_fill()


template<typename System, typename ForwardIterator, typename Size, typename T>
  ForwardIterator strip_const_uninitialized_fill_n(const System &system,
                                                   ForwardIterator first,
                                                   Size n,
                                                   const T &x)
{
  System &non_const_system = const_cast<System&>(system);
  return thrust::uninitialized_fill_n(non_const_system, first, n, x);
} // end strip_const_uninitialized_fill_n()


} // end detail


template<typename ForwardIterator,
         typename T>
  void uninitialized_fill(ForwardIterator first,
                          ForwardIterator last,
                          const T &x)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<ForwardIterator>::type system;

  thrust::detail::strip_const_uninitialized_fill(select_system(system()), first, last, x);
} // end uninitialized_fill()


template<typename ForwardIterator,
         typename Size,
         typename T>
  ForwardIterator uninitialized_fill_n(ForwardIterator first,
                                       Size n,
                                       const T &x)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<ForwardIterator>::type system;

  return thrust::detail::strip_const_uninitialized_fill_n(select_system(system()), first, n, x);
} // end uninitialized_fill_n()


} // end thrust

