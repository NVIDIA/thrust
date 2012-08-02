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


/*! \file remove.inl
 *  \brief Inline file for remove.h.
 */

#include <thrust/detail/config.h>
#include <thrust/remove.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/system/detail/generic/select_system.h>
#include <thrust/system/detail/generic/remove.h>
#include <thrust/system/detail/adl/remove.h>

namespace thrust
{


template<typename System,
         typename ForwardIterator,
         typename T>
  ForwardIterator remove(thrust::detail::dispatchable_base<System> &system,
                         ForwardIterator first,
                         ForwardIterator last,
                         const T &value)
{
  using thrust::system::detail::generic::remove;
  return remove(system.derived(), first, last, value);
} // end remove()


template<typename System,
         typename InputIterator,
         typename OutputIterator,
         typename T>
  OutputIterator remove_copy(thrust::detail::dispatchable_base<System> &system,
                             InputIterator first,
                             InputIterator last,
                             OutputIterator result,
                             const T &value)
{
  using thrust::system::detail::generic::remove_copy;
  return remove_copy(system.derived(), first, last, result, value);
} // end remove_copy()


template<typename System,
         typename ForwardIterator,
         typename Predicate>
  ForwardIterator remove_if(thrust::detail::dispatchable_base<System> &system,
                            ForwardIterator first,
                            ForwardIterator last,
                            Predicate pred)
{
  using thrust::system::detail::generic::remove_if;
  return remove_if(system.derived(), first, last, pred);
} // end remove_if()


template<typename System,
         typename InputIterator,
         typename OutputIterator,
         typename Predicate>
  OutputIterator remove_copy_if(thrust::detail::dispatchable_base<System> &system,
                                InputIterator first,
                                InputIterator last,
                                OutputIterator result,
                                Predicate pred)
{
  using thrust::system::detail::generic::remove_copy_if;
  return remove_copy_if(system.derived(), first, last, result, pred);
} // end remove_copy_if()


template<typename System,
         typename ForwardIterator,
         typename InputIterator,
         typename Predicate>
  ForwardIterator remove_if(thrust::detail::dispatchable_base<System> &system,
                            ForwardIterator first,
                            ForwardIterator last,
                            InputIterator stencil,
                            Predicate pred)
{
  using thrust::system::detail::generic::remove_if;
  return remove_if(system.derived(), first, last, stencil, pred);
} // end remove_if()


template<typename System,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename Predicate>
  OutputIterator remove_copy_if(thrust::detail::dispatchable_base<System> &system,
                                InputIterator1 first,
                                InputIterator1 last,
                                InputIterator2 stencil,
                                OutputIterator result,
                                Predicate pred)
{
  using thrust::system::detail::generic::remove_copy_if;
  return remove_copy_if(system.derived(), first, last, stencil, result, pred);
} // end remove_copy_if()


namespace detail
{


template<typename System,
         typename ForwardIterator,
         typename T>
  ForwardIterator strip_const_remove(const System &system,
                                     ForwardIterator first,
                                     ForwardIterator last,
                                     const T &value)
{
  System &non_const_system = const_cast<System&>(system);
  return thrust::remove(non_const_system, first, last, value);
} // end remove()


template<typename System,
         typename InputIterator,
         typename OutputIterator,
         typename T>
  OutputIterator strip_const_remove_copy(const System &system,
                                         InputIterator first,
                                         InputIterator last,
                                         OutputIterator result,
                                         const T &value)
{
  System &non_const_system = const_cast<System&>(system);
  return thrust::remove_copy(non_const_system, first, last, result, value);
} // end remove_copy()


template<typename System,
         typename ForwardIterator,
         typename Predicate>
  ForwardIterator strip_const_remove_if(const System &system,
                                        ForwardIterator first,
                                        ForwardIterator last,
                                        Predicate pred)
{
  System &non_const_system = const_cast<System&>(system);
  return thrust::remove_if(non_const_system, first, last, pred);
} // end remove_if()


template<typename System,
         typename InputIterator,
         typename OutputIterator,
         typename Predicate>
  OutputIterator strip_const_remove_copy_if(const System &system,
                                            InputIterator first,
                                            InputIterator last,
                                            OutputIterator result,
                                            Predicate pred)
{
  System &non_const_system = const_cast<System&>(system);
  return thrust::remove_copy_if(non_const_system, first, last, result, pred);
} // end remove_copy_if()


template<typename System,
         typename ForwardIterator,
         typename InputIterator,
         typename Predicate>
  ForwardIterator strip_const_remove_if(const System &system,
                                        ForwardIterator first,
                                        ForwardIterator last,
                                        InputIterator stencil,
                                        Predicate pred)
{
  System &non_const_system = const_cast<System&>(system);
  return thrust::remove_if(non_const_system, first, last, stencil, pred);
} // end remove_if()


template<typename System,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename Predicate>
  OutputIterator strip_const_remove_copy_if(const System &system,
                                            InputIterator1 first,
                                            InputIterator1 last,
                                            InputIterator2 stencil,
                                            OutputIterator result,
                                            Predicate pred)
{
  System &non_const_system = const_cast<System&>(system);
  return thrust::remove_copy_if(non_const_system, first, last, stencil, result, pred);
} // end remove_copy_if()


} // end detail


template<typename ForwardIterator,
         typename T>
  ForwardIterator remove(ForwardIterator first,
                         ForwardIterator last,
                         const T &value)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<ForwardIterator>::type System;

  System system;

  return thrust::detail::strip_const_remove(select_system(system), first, last, value);
} // end remove()


template<typename InputIterator,
         typename OutputIterator,
         typename T>
  OutputIterator remove_copy(InputIterator first,
                             InputIterator last,
                             OutputIterator result,
                             const T &value)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<InputIterator>::type  System1;
  typedef typename thrust::iterator_system<OutputIterator>::type System2;

  System1 system1;
  System2 system2;

  return thrust::detail::strip_const_remove_copy(select_system(system1,system2), first, last, result, value);
} // end remove_copy()


template<typename ForwardIterator,
         typename Predicate>
  ForwardIterator remove_if(ForwardIterator first,
                            ForwardIterator last,
                            Predicate pred)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<ForwardIterator>::type System;

  System system;

  return thrust::detail::strip_const_remove_if(select_system(system), first, last, pred);
} // end remove_if()


template<typename ForwardIterator,
         typename InputIterator,
         typename Predicate>
  ForwardIterator remove_if(ForwardIterator first,
                            ForwardIterator last,
                            InputIterator stencil,
                            Predicate pred)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<ForwardIterator>::type System1;
  typedef typename thrust::iterator_system<InputIterator>::type   System2;

  System1 system1;
  System2 system2;

  return thrust::detail::strip_const_remove_if(select_system(system1,system2), first, last, stencil, pred);
} // end remove_if()


template<typename InputIterator,
         typename OutputIterator,
         typename Predicate>
  OutputIterator remove_copy_if(InputIterator first,
                                InputIterator last,
                                OutputIterator result,
                                Predicate pred)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<InputIterator>::type  System1;
  typedef typename thrust::iterator_system<OutputIterator>::type System2;

  System1 system1;
  System2 system2;

  return thrust::detail::strip_const_remove_copy_if(select_system(system1,system2), first, last, result, pred);
} // end remove_copy_if()


template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename Predicate>
  OutputIterator remove_copy_if(InputIterator1 first,
                                InputIterator1 last,
                                InputIterator2 stencil,
                                OutputIterator result,
                                Predicate pred)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<InputIterator1>::type System1;
  typedef typename thrust::iterator_system<InputIterator2>::type System2;
  typedef typename thrust::iterator_system<OutputIterator>::type System3;

  System1 system1;
  System2 system2;
  System3 system3;

  return thrust::detail::strip_const_remove_copy_if(select_system(system1,system2,system3), first, last, stencil, result, pred);
} // end remove_copy_if()


} // end namespace thrust

