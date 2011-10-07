/*
 *  Copyright 2008-2011 NVIDIA Corporation
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
#include <thrust/detail/backend/generic/select_system.h>
#include <thrust/detail/backend/generic/transform.h>
#include <thrust/iterator/iterator_traits.h>

namespace thrust
{

template<typename InputIterator,
         typename OutputIterator,
         typename UnaryFunction>
  OutputIterator transform(InputIterator first,
                           InputIterator last,
                           OutputIterator result,
                           UnaryFunction op)
{
  using thrust::detail::backend::generic::select_system;
  using thrust::detail::backend::generic::transform;

  typedef typename thrust::iterator_space<InputIterator>::type  space1;
  typedef typename thrust::iterator_space<OutputIterator>::type space2;

  return transform(select_system(space1(),space2()), first, last, result, op);
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
  using thrust::detail::backend::generic::select_system;
  using thrust::detail::backend::generic::transform;

  typedef typename thrust::iterator_space<InputIterator1>::type space1;
  typedef typename thrust::iterator_space<InputIterator2>::type space2;
  typedef typename thrust::iterator_space<OutputIterator>::type space3;

  return transform(select_system(space1(),space2(),space3()), first1, last1, first2, result, op);
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
  using thrust::detail::backend::generic::select_system;
  using thrust::detail::backend::generic::transform_if;

  typedef typename thrust::iterator_space<InputIterator>::type   space1;
  typedef typename thrust::iterator_space<ForwardIterator>::type space2;

  return transform_if(select_system(space1(),space2()), first, last, result, unary_op, pred);
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
  using thrust::detail::backend::generic::select_system;
  using thrust::detail::backend::generic::transform_if;

  typedef typename thrust::iterator_space<InputIterator1>::type  space1;
  typedef typename thrust::iterator_space<InputIterator2>::type  space2;
  typedef typename thrust::iterator_space<ForwardIterator>::type space3;

  return transform_if(select_system(space1(),space2(),space3()), first, last, stencil, result, unary_op, pred);
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
  using thrust::detail::backend::generic::select_system;
  using thrust::detail::backend::generic::transform_if;

  typedef typename thrust::iterator_space<InputIterator1>::type  space1;
  typedef typename thrust::iterator_space<InputIterator2>::type  space2;
  typedef typename thrust::iterator_space<InputIterator3>::type  space3;
  typedef typename thrust::iterator_space<ForwardIterator>::type space4;

  return transform_if(select_system(space1(),space2(),space3(),space4()), first1, last1, first2, stencil, result, binary_op, pred);
} // end transform_if()

} // end namespace thrust

