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

  typedef typename thrust::iterator_space<InputIterator1>::type       space1; 
  typedef typename thrust::iterator_space<InputIterator2>::type       space2; 
  typedef typename thrust::iterator_space<RandomAccessIterator>::type space3; 

  return scatter(select_system(space1(),space2(),space3()), first, last, map, output);
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

  typedef typename thrust::iterator_space<InputIterator1>::type       space1; 
  typedef typename thrust::iterator_space<InputIterator2>::type       space2; 
  typedef typename thrust::iterator_space<InputIterator3>::type       space3; 
  typedef typename thrust::iterator_space<RandomAccessIterator>::type space4; 

  return scatter_if(select_system(space1(),space2(),space3()), first, last, map, stencil, output);
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

  typedef typename thrust::iterator_space<InputIterator1>::type       space1; 
  typedef typename thrust::iterator_space<InputIterator2>::type       space2; 
  typedef typename thrust::iterator_space<InputIterator3>::type       space3; 
  typedef typename thrust::iterator_space<RandomAccessIterator>::type space4; 

  return scatter_if(select_system(space1(),space2(),space3()), first, last, map, stencil, output, pred);
} // end scatter_if()

} // end namespace thrust

