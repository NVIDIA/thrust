/*
 *  Copyright 2008-2010 NVIDIA Corporation
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
 *  \brief Inline file for transform.h
 */

#pragma once

#include <thrust/detail/device/for_each.h>
#include <thrust/distance.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

namespace thrust
{
namespace detail
{
namespace device
{

// WAR circular #inclusion with forward declaration of for_each 
template<typename InputIterator, typename UnaryFunction> void for_each(InputIterator, InputIterator, UnaryFunction);

namespace generic
{
namespace detail
{

template <typename UnaryFunction>
struct unary_transform_functor
{
  UnaryFunction f;

  unary_transform_functor(UnaryFunction f_)
    :f(f_)
  {}

  template<typename Tuple>
  __host__ __device__
  void operator()(Tuple t)
  {
    thrust::get<1>(t) = f(thrust::get<0>(t));
  }
}; // end unary_transform_functor


template <typename BinaryFunction>
struct binary_transform_functor
{
  BinaryFunction f;

  binary_transform_functor(BinaryFunction f_)
    :f(f_)
  {}

  template <typename Tuple>
  __host__ __device__
  void operator()(Tuple t)
  { 
    thrust::get<2>(t) = f(thrust::get<0>(t), thrust::get<1>(t));
  }
}; // end binary_transform_functor


template <typename UnaryFunction, typename Predicate>
struct unary_transform_if_functor
{
  UnaryFunction unary_op;
  Predicate pred;
  
  unary_transform_if_functor(UnaryFunction _unary_op, Predicate _pred)
    : unary_op(_unary_op), pred(_pred) {} 
  
  template <typename Tuple>
  __host__ __device__
  void operator()(Tuple t)
  {
    if(pred(thrust::get<1>(t)))
      thrust::get<2>(t) = unary_op(thrust::get<0>(t));
  }
}; // end unary_transform_if_functor


template <typename BinaryFunction, typename Predicate>
struct binary_transform_if_functor
{
  BinaryFunction binary_op;
  Predicate pred;

  binary_transform_if_functor(BinaryFunction _binary_op, Predicate _pred)
    : binary_op(_binary_op), pred(_pred) {} 

  template <typename Tuple>
  __host__ __device__
  void operator()(Tuple t)
  {
    if(pred(thrust::get<2>(t)))
      thrust::get<3>(t) = binary_op(thrust::get<0>(t), thrust::get<1>(t));
  }
}; // end binary_transform_if_functor

} // end namespace detail


template<typename InputIterator,
         typename OutputIterator,
         typename UnaryFunction>
  OutputIterator transform(InputIterator first, InputIterator last,
                           OutputIterator result,
                           UnaryFunction unary_op)
{
  detail::unary_transform_functor<UnaryFunction> func(unary_op);
  
  thrust::detail::device::for_each(thrust::make_zip_iterator(thrust::make_tuple(first, result)),
                                   thrust::make_zip_iterator(thrust::make_tuple(first, result)) + thrust::distance(first, last),
                                   func);
  
  return result + (last - first); // return the end of the output sequence
} // end transform() 


template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename BinaryFunction>
  OutputIterator transform(InputIterator1 first1, InputIterator1 last1,
                           InputIterator2 first2,
                           OutputIterator result,
                           BinaryFunction binary_op)
{
  detail::binary_transform_functor<BinaryFunction> func(binary_op);
  
  thrust::detail::device::for_each(thrust::make_zip_iterator(thrust::make_tuple(first1, first2, result)),
                                   thrust::make_zip_iterator(thrust::make_tuple(first1, first2, result)) + thrust::distance(first1, last1),
                                   func);
  
  return result + (last1 - first1); // return the end of the output sequence
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
  detail::unary_transform_if_functor<UnaryFunction,Predicate> func(unary_op, pred);

  thrust::detail::device::for_each(thrust::make_zip_iterator(thrust::make_tuple(first, stencil, result)),
                                   thrust::make_zip_iterator(thrust::make_tuple(first, stencil, result)) + thrust::distance(first, last),
                                   func);

  return result + (last - first); // return the end of the output sequence
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
  detail::binary_transform_if_functor<BinaryFunction,Predicate> func(binary_op, pred);

  thrust::detail::device::for_each(thrust::make_zip_iterator(thrust::make_tuple(first1, first2, stencil, result)),
                                   thrust::make_zip_iterator(thrust::make_tuple(first1, first2, stencil, result)) + thrust::distance(first1, last1),
                                   func);

  return result + (last1 - first1); // return the end of the output sequence
} // end transform_if()

} // end namespace generic
} // end namespace device
} // end namespace detail
} // end namespace thrust

