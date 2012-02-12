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
#include <thrust/detail/adl_helper.h>

namespace thrust
{

template<typename InputIterator1,
         typename InputIterator2,
         typename RandomAccessIterator>
  void scatter(InputIterator1 first,
               InputIterator1 last,
               InputIterator2 map,
               RandomAccessIterator output)
{
  using thrust::system::detail::generic::select_system;
  using thrust::system::detail::generic::scatter;

  typedef typename thrust::iterator_system<InputIterator1>::type       system1; 
  typedef typename thrust::iterator_system<InputIterator2>::type       system2; 
  typedef typename thrust::iterator_system<RandomAccessIterator>::type system3; 

  return scatter(select_system(system1(),system2(),system3()), first, last, map, output);
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
  using thrust::system::detail::generic::scatter_if;

  typedef typename thrust::iterator_system<InputIterator1>::type       system1; 
  typedef typename thrust::iterator_system<InputIterator2>::type       system2; 
  typedef typename thrust::iterator_system<InputIterator3>::type       system3; 
  typedef typename thrust::iterator_system<RandomAccessIterator>::type system4; 

  return scatter_if(select_system(system1(),system2(),system3()), first, last, map, stencil, output);
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
  using thrust::system::detail::generic::scatter_if;

  typedef typename thrust::iterator_system<InputIterator1>::type       system1; 
  typedef typename thrust::iterator_system<InputIterator2>::type       system2; 
  typedef typename thrust::iterator_system<InputIterator3>::type       system3; 
  typedef typename thrust::iterator_system<RandomAccessIterator>::type system4; 

  return scatter_if(select_system(system1(),system2(),system3()), first, last, map, stencil, output, pred);
} // end scatter_if()

} // end namespace thrust

