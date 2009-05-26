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


/*! \file transform.inl
 *  \brief Inline file for transform.h.
 */

#include <thrust/transform.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/detail/dispatch/transform.h>

namespace thrust
{

template<typename InputIterator,
         typename OutputIterator,
         typename UnaryFunction>
  OutputIterator transform(InputIterator first, InputIterator last,
                           OutputIterator result,
                           UnaryFunction op)
{
    // dispatch on category
    return thrust::detail::dispatch::transform(first, last, result, op,
            typename thrust::iterator_traits<InputIterator>::iterator_category(),
            typename thrust::iterator_traits<OutputIterator>::iterator_category());
} // end transform()


template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename BinaryFunction>
  OutputIterator transform(InputIterator1 first1, InputIterator1 last1,
                           InputIterator2 first2,
                           OutputIterator result,
                           BinaryFunction op)
{
    // dispatch on category
    return thrust::detail::dispatch::transform(first1, last1, first2, result, op,
            typename thrust::iterator_traits<InputIterator1>::iterator_category(),
            typename thrust::iterator_traits<InputIterator2>::iterator_category(),
            typename thrust::iterator_traits<OutputIterator>::iterator_category());
} // end transform()


template<typename InputIterator1,
         typename InputIterator2,
         typename ForwardIterator,
         typename UnaryFunction,
         typename Predicate>
  ForwardIterator transform_if(InputIterator1 first, InputIterator1 last,
                               InputIterator2 stencil,
                               ForwardIterator result,
                               UnaryFunction unary_op,
                               Predicate pred)
{
    // dispatch on category
    return thrust::detail::dispatch::transform_if(first, last, stencil, result, unary_op, pred,
            typename thrust::iterator_traits<InputIterator1>::iterator_category(),
            typename thrust::iterator_traits<InputIterator2>::iterator_category(),
            typename thrust::iterator_traits<ForwardIterator>::iterator_category());
} // end transform_if()

template<typename InputIterator1,
         typename InputIterator2,
         typename InputIterator3,
         typename ForwardIterator,
         typename BinaryFunction,
         typename Predicate>
  ForwardIterator transform_if(InputIterator1 first1, InputIterator1 last1,
                               InputIterator2 first2,
                               InputIterator3 stencil,
                               ForwardIterator result,
                               BinaryFunction binary_op,
                               Predicate pred)
{
    // dispatch on category
    return thrust::detail::dispatch::transform_if(first1, last1, first2, stencil, result, binary_op, pred,
            typename thrust::iterator_traits<InputIterator1>::iterator_category(),
            typename thrust::iterator_traits<InputIterator2>::iterator_category(),
            typename thrust::iterator_traits<InputIterator3>::iterator_category(),
            typename thrust::iterator_traits<ForwardIterator>::iterator_category());
} // end transform_if()

} // end namespace thrust

