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

#include <thrust/detail/config.h>
#include <thrust/tabulate.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/system/detail/generic/select_system.h>
#include <thrust/system/detail/generic/tabulate.h>
#include <thrust/system/detail/adl/tabulate.h>

namespace thrust
{


template<typename System, typename ForwardIterator, typename UnaryOperation>
  void tabulate(thrust::detail::dispatchable_base<System> &system,
                ForwardIterator first,
                ForwardIterator last,
                UnaryOperation unary_op)
{
  using thrust::system::detail::generic::tabulate;
  return tabulate(system.derived(), first, last, unary_op);
} // end tabulate()


namespace detail
{


template<typename System, typename ForwardIterator, typename UnaryOperation>
  void strip_const_tabulate(const System &system,
                            ForwardIterator first,
                            ForwardIterator last,
                            UnaryOperation unary_op)
{
  System &non_const_system = const_cast<System&>(system);
  return thrust::tabulate(non_const_system, first, last, unary_op);
} // end tabulate()


} // end detail


template<typename ForwardIterator, typename UnaryOperation>
  void tabulate(ForwardIterator first,
                ForwardIterator last,
                UnaryOperation unary_op)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<ForwardIterator>::type System;

  System system;

  return thrust::detail::strip_const_tabulate(select_system(system), first, last, unary_op);
} // end tabulate()


} // end namespace thrust

