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


/*! \file transform.inl
 *  \brief Inline file for transform.h.
 */

#include <thrust/transform.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/system/detail/generic/select_system.h>
#include <thrust/system/detail/generic/transform.h>
#include <thrust/system/detail/adl/transform.h>

namespace thrust
{


template<typename System,
         typename InputIterator,
         typename OutputIterator,
         typename UnaryFunction>
  OutputIterator transform(thrust::detail::dispatchable_base<System> &system,
                           InputIterator first, InputIterator last,
                           OutputIterator result,
                           UnaryFunction op)
{
  using thrust::system::detail::generic::transform;
  return transform(system.derived(), first, last, result, op);
} // end transform()


template<typename System,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename BinaryFunction>
  OutputIterator transform(thrust::detail::dispatchable_base<System> &system,
                           InputIterator1 first1, InputIterator1 last1,
                           InputIterator2 first2,
                           OutputIterator result,
                           BinaryFunction op)
{
  using thrust::system::detail::generic::transform;
  return transform(system.derived(), first1, last1, first2, result, op);
} // end transform()


template<typename System,
         typename InputIterator,
         typename ForwardIterator,
         typename UnaryFunction,
         typename Predicate>
  ForwardIterator transform_if(thrust::detail::dispatchable_base<System> &system,
                               InputIterator first, InputIterator last,
                               ForwardIterator result,
                               UnaryFunction op,
                               Predicate pred)
{
  using thrust::system::detail::generic::transform_if;
  return transform_if(system.derived(), first, last, result, op, pred);
} // end transform_if()


template<typename System,
         typename InputIterator1,
         typename InputIterator2,
         typename ForwardIterator,
         typename UnaryFunction,
         typename Predicate>
  ForwardIterator transform_if(thrust::detail::dispatchable_base<System> &system,
                               InputIterator1 first, InputIterator1 last,
                               InputIterator2 stencil,
                               ForwardIterator result,
                               UnaryFunction op,
                               Predicate pred)
{
  using thrust::system::detail::generic::transform_if;
  return transform_if(system.derived(), first, last, stencil, result, op, pred);
} // end transform_if()


template<typename System,
         typename InputIterator1,
         typename InputIterator2,
         typename InputIterator3,
         typename ForwardIterator,
         typename BinaryFunction,
         typename Predicate>
  ForwardIterator transform_if(thrust::detail::dispatchable_base<System> &system,
                               InputIterator1 first1, InputIterator1 last1,
                               InputIterator2 first2,
                               InputIterator3 stencil,
                               ForwardIterator result,
                               BinaryFunction binary_op,
                               Predicate pred)
{
  using thrust::system::detail::generic::transform_if;
  return transform_if(system.derived(), first1, last1, first2, stencil, result, binary_op, pred);
} // end transform_if()


namespace detail
{


template<typename System,
         typename InputIterator,
         typename OutputIterator,
         typename UnaryFunction>
  OutputIterator strip_const_transform(const System &system,
                                       InputIterator first, InputIterator last,
                                       OutputIterator result,
                                       UnaryFunction op)
{
  System &non_const_system = const_cast<System&>(system);
  return thrust::transform(non_const_system, first, last, result, op);
} // end transform()


template<typename System,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename BinaryFunction>
  OutputIterator strip_const_transform(const System &system,
                                       InputIterator1 first1, InputIterator1 last1,
                                       InputIterator2 first2,
                                       OutputIterator result,
                                       BinaryFunction op)
{
  System &non_const_system = const_cast<System&>(system);
  return thrust::transform(non_const_system, first1, last1, first2, result, op);
} // end transform()


template<typename System,
         typename InputIterator,
         typename ForwardIterator,
         typename UnaryFunction,
         typename Predicate>
  ForwardIterator strip_const_transform_if(const System &system,
                                           InputIterator first, InputIterator last,
                                           ForwardIterator result,
                                           UnaryFunction op,
                                           Predicate pred)
{
  System &non_const_system = const_cast<System&>(system);
  return thrust::transform_if(non_const_system, first, last, result, op, pred);
} // end transform_if()


template<typename System,
         typename InputIterator1,
         typename InputIterator2,
         typename ForwardIterator,
         typename UnaryFunction,
         typename Predicate>
  ForwardIterator strip_const_transform_if(const System &system,
                                           InputIterator1 first, InputIterator1 last,
                                           InputIterator2 stencil,
                                           ForwardIterator result,
                                           UnaryFunction op,
                                           Predicate pred)
{
  System &non_const_system = const_cast<System&>(system);
  return thrust::transform_if(non_const_system, first, last, stencil, result, op, pred);
} // end transform_if()


template<typename System,
         typename InputIterator1,
         typename InputIterator2,
         typename InputIterator3,
         typename ForwardIterator,
         typename BinaryFunction,
         typename Predicate>
  ForwardIterator strip_const_transform_if(const System &system,
                                           InputIterator1 first1, InputIterator1 last1,
                                           InputIterator2 first2,
                                           InputIterator3 stencil,
                                           ForwardIterator result,
                                           BinaryFunction binary_op,
                                           Predicate pred)
{
  System &non_const_system = const_cast<System&>(system);
  return thrust::transform_if(non_const_system, first1, last1, first2, stencil, result, binary_op, pred);
} // end transform_if()


} // end detail


template<typename InputIterator,
         typename OutputIterator,
         typename UnaryFunction>
  OutputIterator transform(InputIterator first,
                           InputIterator last,
                           OutputIterator result,
                           UnaryFunction op)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<InputIterator>::type  System1;
  typedef typename thrust::iterator_system<OutputIterator>::type System2;

  System1 system1;
  System2 system2;

  return thrust::detail::strip_const_transform(select_system(system1,system2), first, last, result, op);
} // end transform()


template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename BinaryFunction>
  OutputIterator transform(InputIterator1 first1,
                           InputIterator1 last1,
                           InputIterator2 first2,
                           OutputIterator result,
                           BinaryFunction op)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<InputIterator1>::type System1;
  typedef typename thrust::iterator_system<InputIterator2>::type System2;
  typedef typename thrust::iterator_system<OutputIterator>::type System3;

  System1 system1;
  System2 system2;
  System3 system3;

  return thrust::detail::strip_const_transform(select_system(system1,system2,system3), first1, last1, first2, result, op);
} // end transform()


template<typename InputIterator,
         typename ForwardIterator,
         typename UnaryFunction,
         typename Predicate>
  ForwardIterator transform_if(InputIterator first,
                               InputIterator last,
                               ForwardIterator result,
                               UnaryFunction unary_op,
                               Predicate pred)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<InputIterator>::type   System1;
  typedef typename thrust::iterator_system<ForwardIterator>::type System2;

  System1 system1;
  System2 system2;

  return thrust::detail::strip_const_transform_if(select_system(system1,system2), first, last, result, unary_op, pred);
} // end transform_if()


template<typename InputIterator1,
         typename InputIterator2,
         typename ForwardIterator,
         typename UnaryFunction,
         typename Predicate>
  ForwardIterator transform_if(InputIterator1 first,
                               InputIterator1 last,
                               InputIterator2 stencil,
                               ForwardIterator result,
                               UnaryFunction unary_op,
                               Predicate pred)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<InputIterator1>::type  System1;
  typedef typename thrust::iterator_system<InputIterator2>::type  System2;
  typedef typename thrust::iterator_system<ForwardIterator>::type System3;

  System1 system1;
  System2 system2;
  System3 system3;

  return thrust::detail::strip_const_transform_if(select_system(system1,system2,system3), first, last, stencil, result, unary_op, pred);
} // end transform_if()


template<typename InputIterator1,
         typename InputIterator2,
         typename InputIterator3,
         typename ForwardIterator,
         typename BinaryFunction,
         typename Predicate>
  ForwardIterator transform_if(InputIterator1 first1,
                               InputIterator1 last1,
                               InputIterator2 first2,
                               InputIterator3 stencil,
                               ForwardIterator result,
                               BinaryFunction binary_op,
                               Predicate pred)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<InputIterator1>::type  System1;
  typedef typename thrust::iterator_system<InputIterator2>::type  System2;
  typedef typename thrust::iterator_system<InputIterator3>::type  System3;
  typedef typename thrust::iterator_system<ForwardIterator>::type System4;

  System1 system1;
  System2 system2;
  System3 system3;
  System4 system4;

  return thrust::detail::strip_const_transform_if(select_system(system1,system2,system3,system4), first1, last1, first2, stencil, result, binary_op, pred);
} // end transform_if()


} // end namespace thrust

