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


/*! \file gather.inl
 *  \brief Inline file for gather.h.
 */

#include <thrust/gather.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/detail/dispatch/gather.h>

namespace thrust
{

template<typename ForwardIterator,
         typename InputIterator,
         typename RandomAccessIterator>
  void gather(ForwardIterator first,
              ForwardIterator last,
              InputIterator map,
              RandomAccessIterator input)
{
  // dispatch on category
  thrust::detail::dispatch::gather(first, last, map, input,
          typename thrust::iterator_traits<ForwardIterator>::iterator_category(),
          typename thrust::iterator_traits<InputIterator>::iterator_category(),
          typename thrust::iterator_traits<RandomAccessIterator>::iterator_category());
} // end gather()

template<typename ForwardIterator,
         typename InputIterator1,
         typename InputIterator2,
         typename RandomAccessIterator>
  void gather_if(ForwardIterator first,
                 ForwardIterator last,
                 InputIterator1 map,
                 InputIterator2 stencil,
                 RandomAccessIterator input)
{
  typedef typename thrust::iterator_traits<InputIterator2>::value_type StencilType;
  thrust::gather_if(first, last, map, stencil, input, thrust::identity<StencilType>());
} // end gather_if()

template<typename ForwardIterator,
         typename InputIterator1,
         typename InputIterator2,
         typename RandomAccessIterator,
         typename Predicate>
  void gather_if(ForwardIterator first,
                 ForwardIterator last,
                 InputIterator1 map,
                 InputIterator2 stencil,
                 RandomAccessIterator input,
                 Predicate pred)
{
  // dispatch on category
  thrust::detail::dispatch::gather_if(first, last, map, stencil, input, pred,
          typename thrust::iterator_traits<ForwardIterator>::iterator_category(),
          typename thrust::iterator_traits<InputIterator1>::iterator_category(),
          typename thrust::iterator_traits<InputIterator2>::iterator_category(),
          typename thrust::iterator_traits<RandomAccessIterator>::iterator_category());
} // end gather_if()

} // end thrust

