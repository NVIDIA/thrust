/*
 *  Copyright 2008-2009 NVIDIA Corporation
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
#include <thrust/detail/dispatch/scatter.h>

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
  // dispatch on category
  thrust::detail::dispatch::scatter(first, last, map, output,
    typename thrust::iterator_traits<InputIterator1>::iterator_category(),
    typename thrust::iterator_traits<InputIterator2>::iterator_category(),
    typename thrust::iterator_traits<RandomAccessIterator>::iterator_category());
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
  // default predicate is identity
  typedef typename thrust::iterator_traits<InputIterator3>::value_type StencilType;
  scatter_if(first, last, map, stencil, output, thrust::identity<StencilType>());
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
  // dispatch on category
  thrust::detail::dispatch::scatter_if(first, last, map, stencil, output, pred,
    typename thrust::iterator_traits<InputIterator1>::iterator_category(),
    typename thrust::iterator_traits<InputIterator2>::iterator_category(),
    typename thrust::iterator_traits<InputIterator3>::iterator_category(),
    typename thrust::iterator_traits<RandomAccessIterator>::iterator_category());
} // end scatter_if()

} // end namespace thrust

