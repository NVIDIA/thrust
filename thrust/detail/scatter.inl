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


/*! \file scatter.inl
 *  \brief Inline file for scatter.h.
 */

#include <thrust/scatter.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/system/detail/generic/select_system.h>
#include <thrust/system/detail/generic/scatter.h>
#include <thrust/system/detail/adl/scatter.h>

namespace thrust
{


template<typename System,
         typename InputIterator1,
         typename InputIterator2,
         typename RandomAccessIterator>
  void scatter(thrust::detail::dispatchable_base<System> &system,
               InputIterator1 first,
               InputIterator1 last,
               InputIterator2 map,
               RandomAccessIterator output)
{
  using thrust::system::detail::generic::scatter;
  return scatter(system.derived(), first, last, map, output);
} // end scatter()


template<typename System,
         typename InputIterator1,
         typename InputIterator2,
         typename InputIterator3,
         typename RandomAccessIterator>
  void scatter_if(thrust::detail::dispatchable_base<System> &system,
                  InputIterator1 first,
                  InputIterator1 last,
                  InputIterator2 map,
                  InputIterator3 stencil,
                  RandomAccessIterator output)
{
  using thrust::system::detail::generic::scatter_if;
  return scatter_if(system.derived(), first, last, map, stencil, output);
} // end scatter_if()


template<typename System,
         typename InputIterator1,
         typename InputIterator2,
         typename InputIterator3,
         typename RandomAccessIterator,
         typename Predicate>
  void scatter_if(thrust::detail::dispatchable_base<System> &system,
                  InputIterator1 first,
                  InputIterator1 last,
                  InputIterator2 map,
                  InputIterator3 stencil,
                  RandomAccessIterator output,
                  Predicate pred)
{
  using thrust::system::detail::generic::scatter_if;
  return scatter_if(system.derived(), first, last, map, stencil, output, pred);
} // end scatter_if()


namespace detail
{


template<typename System,
         typename InputIterator1,
         typename InputIterator2,
         typename RandomAccessIterator>
  void strip_const_scatter(const System &system,
                           InputIterator1 first,
                           InputIterator1 last,
                           InputIterator2 map,
                           RandomAccessIterator output)
{
  System &non_const_system = const_cast<System&>(system);
  return thrust::scatter(non_const_system, first, last, map, output);
} // end scatter()


template<typename System,
         typename InputIterator1,
         typename InputIterator2,
         typename InputIterator3,
         typename RandomAccessIterator>
  void strip_const_scatter_if(const System &system,
                              InputIterator1 first,
                              InputIterator1 last,
                              InputIterator2 map,
                              InputIterator3 stencil,
                              RandomAccessIterator output)
{
  System &non_const_system = const_cast<System&>(system);
  return thrust::scatter_if(non_const_system, first, last, map, stencil, output);
} // end scatter_if()


template<typename System,
         typename InputIterator1,
         typename InputIterator2,
         typename InputIterator3,
         typename RandomAccessIterator,
         typename Predicate>
  void strip_const_scatter_if(const System &system,
                              InputIterator1 first,
                              InputIterator1 last,
                              InputIterator2 map,
                              InputIterator3 stencil,
                              RandomAccessIterator output,
                              Predicate pred)
{
  System &non_const_system = const_cast<System&>(system);
  return thrust::scatter_if(non_const_system, first, last, map, stencil, output, pred);
} // end scatter_if()


} // end detail


template<typename InputIterator1,
         typename InputIterator2,
         typename RandomAccessIterator>
  void scatter(InputIterator1 first,
               InputIterator1 last,
               InputIterator2 map,
               RandomAccessIterator output)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<InputIterator1>::type       System1; 
  typedef typename thrust::iterator_system<InputIterator2>::type       System2; 
  typedef typename thrust::iterator_system<RandomAccessIterator>::type System3; 

  System1 system1;
  System2 system2;
  System3 system3;

  return thrust::detail::strip_const_scatter(select_system(system1,system2,system3), first, last, map, output);
} // end scatter()


template<typename InputIterator1,
         typename InputIterator2,
         typename InputIterator3,
         typename RandomAccessIterator>
  void scatter_if(InputIterator1 first,
                  InputIterator1 last,
                  InputIterator2 map,
                  InputIterator3 stencil,
                  RandomAccessIterator output)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<InputIterator1>::type       System1; 
  typedef typename thrust::iterator_system<InputIterator2>::type       System2; 
  typedef typename thrust::iterator_system<InputIterator3>::type       System3; 
  typedef typename thrust::iterator_system<RandomAccessIterator>::type System4; 

  System1 system1;
  System2 system2;
  System3 system3;
  System4 system4;

  return thrust::detail::strip_const_scatter_if(select_system(system1,system2,system3,system4), first, last, map, stencil, output);
} // end scatter_if()


template<typename InputIterator1,
         typename InputIterator2,
         typename InputIterator3,
         typename RandomAccessIterator,
         typename Predicate>
  void scatter_if(InputIterator1 first,
                  InputIterator1 last,
                  InputIterator2 map,
                  InputIterator3 stencil,
                  RandomAccessIterator output,
                  Predicate pred)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<InputIterator1>::type       System1; 
  typedef typename thrust::iterator_system<InputIterator2>::type       System2; 
  typedef typename thrust::iterator_system<InputIterator3>::type       System3; 
  typedef typename thrust::iterator_system<RandomAccessIterator>::type System4; 

  System1 system1;
  System2 system2;
  System3 system3;
  System4 system4;

  return thrust::detail::strip_const_scatter_if(select_system(system1,system2,system3,system4), first, last, map, stencil, output, pred);
} // end scatter_if()

} // end namespace thrust

