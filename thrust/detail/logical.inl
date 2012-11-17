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


/*! \file logical.inl
 *  \brief Inline file for logical.h.
 */

#include <thrust/detail/config.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/system/detail/generic/select_system.h>
#include <thrust/system/detail/generic/logical.h>
#include <thrust/system/detail/adl/logical.h>

namespace thrust
{


template <typename System, typename InputIterator, typename Predicate>
bool all_of(thrust::detail::dispatchable_base<System> &system, InputIterator first, InputIterator last, Predicate pred)
{
  using thrust::system::detail::generic::all_of;
  return all_of(thrust::detail::derived_cast(system), first, last, pred);
} // end all_of()


template <typename System, typename InputIterator, typename Predicate>
bool any_of(thrust::detail::dispatchable_base<System> &system, InputIterator first, InputIterator last, Predicate pred)
{
  using thrust::system::detail::generic::any_of;
  return any_of(thrust::detail::derived_cast(system), first, last, pred);
} // end any_of()


template <typename System, typename InputIterator, typename Predicate>
bool none_of(thrust::detail::dispatchable_base<System> &system, InputIterator first, InputIterator last, Predicate pred)
{
  using thrust::system::detail::generic::none_of;
  return none_of(thrust::detail::derived_cast(system), first, last, pred);
} // end none_of()


namespace detail
{


template <typename System, typename InputIterator, typename Predicate>
bool strip_const_all_of(const System &system, InputIterator first, InputIterator last, Predicate pred)
{
  System &non_const_system = const_cast<System&>(system);
  return thrust::all_of(non_const_system, first, last, pred);
} // end strip_const_all_of()


template <typename System, typename InputIterator, typename Predicate>
bool strip_const_any_of(const System &system, InputIterator first, InputIterator last, Predicate pred)
{
  System &non_const_system = const_cast<System&>(system);
  return thrust::any_of(non_const_system, first, last, pred);
} // end strip_const_any_of()


template <typename System, typename InputIterator, typename Predicate>
bool strip_const_none_of(const System &system, InputIterator first, InputIterator last, Predicate pred)
{
  System &non_const_system = const_cast<System&>(system);
  return thrust::none_of(non_const_system, first, last, pred);
} // end strip_const_none_of()


} // end detail


template <typename InputIterator, typename Predicate>
bool all_of(InputIterator first, InputIterator last, Predicate pred)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<InputIterator>::type System;

  System system;

  return thrust::detail::strip_const_all_of(select_system(system), first, last, pred);
}


template <typename InputIterator, typename Predicate>
bool any_of(InputIterator first, InputIterator last, Predicate pred)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<InputIterator>::type System;

  System system;

  return thrust::detail::strip_const_any_of(select_system(system), first, last, pred);
}


template <typename InputIterator, typename Predicate>
bool none_of(InputIterator first, InputIterator last, Predicate pred)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<InputIterator>::type System;

  System system;

  return thrust::detail::strip_const_none_of(select_system(system), first, last, pred);
}


} // end namespace thrust

