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
#include <thrust/system/detail/generic/transform.h>
#include <thrust/for_each.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/detail/minimum_system.h>
#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/detail/internal_functional.h>

namespace thrust
{
namespace system
{
namespace detail
{
namespace generic
{


template<typename InputIterator,
         typename OutputIterator,
         typename UnaryFunction>
  OutputIterator transform(tag,
                           InputIterator first,
                           InputIterator last,
                           OutputIterator result,
                           UnaryFunction op)
{
  // determine the minimal system of the two iterators
  typedef typename thrust::iterator_system<InputIterator>::type        System1;
  typedef typename thrust::iterator_system<OutputIterator>::type       System2;
  typedef typename thrust::detail::minimum_system<System1,System2>::type System;

  // XXX WAR the problem of a generic __host__ __device__ functor's inability to invoke
  //     a function which is only __host__ or __device__ by selecting a generic functor
  //     which is one or the other
  //     when nvcc is able to deal with this, remove this WAR
  
  // given the minimal system, determine the unary transform functor we need
  typedef typename thrust::detail::unary_transform_functor<System,UnaryFunction>::type UnaryTransformFunctor;

  // make an iterator tuple
  typedef thrust::tuple<InputIterator,OutputIterator> IteratorTuple;
  typedef thrust::zip_iterator<IteratorTuple> ZipIterator;

  ZipIterator zipped_result =
    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(first,result)),
                     thrust::make_zip_iterator(thrust::make_tuple(last,result)),
                     UnaryTransformFunctor(op));

  return thrust::get<1>(zipped_result.get_iterator_tuple());
} // end transform()


template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename BinaryFunction>
  OutputIterator transform(tag,
                           InputIterator1 first1,
                           InputIterator1 last1,
                           InputIterator2 first2,
                           OutputIterator result,
                           BinaryFunction op)
{
  // determine the minimal system of the three iterators
  typedef typename thrust::iterator_system<InputIterator1>::type        System1;
  typedef typename thrust::iterator_system<InputIterator2>::type        System2;
  typedef typename thrust::iterator_system<OutputIterator>::type        System3;

  typedef typename thrust::detail::minimum_system<System1,System2>::type  System4;
  typedef typename thrust::detail::minimum_system<System4,System3>::type  System;

  // XXX WAR the problem of a generic __host__ __device__ functor's inability to invoke
  //     a function which is only __host__ or __device__ by selecting a generic functor
  //     which is one or the other
  //     when nvcc is able to deal with this, remove this WAR
  
  // given the minimal system, determine the binary transform functor we need
  typedef typename thrust::detail::binary_transform_functor<System,BinaryFunction>::type BinaryTransformFunctor;

  // make an iterator tuple
  typedef thrust::tuple<InputIterator1,InputIterator2,OutputIterator> IteratorTuple;
  typedef thrust::zip_iterator<IteratorTuple> ZipIterator;

  ZipIterator zipped_result =
    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(first1,first2,result)),
                     thrust::make_zip_iterator(thrust::make_tuple(last1,first2,result)),
                     BinaryTransformFunctor(op));

  return thrust::get<2>(zipped_result.get_iterator_tuple());
} // end transform()


template<typename InputIterator,
         typename ForwardIterator,
         typename UnaryFunction,
         typename Predicate>
  ForwardIterator transform_if(tag,
                               InputIterator first,
                               InputIterator last,
                               ForwardIterator result,
                               UnaryFunction unary_op,
                               Predicate pred)
{
  // determine the minimal system of the two iterators
  typedef typename thrust::iterator_system<InputIterator>::type        System1;
  typedef typename thrust::iterator_system<ForwardIterator>::type      System2;

  typedef typename thrust::detail::minimum_system<System1,System2>::type System;

  // XXX WAR the problem of a generic __host__ __device__ functor's inability to invoke
  //     a function which is only __host__ or __device__ by selecting a generic functor
  //     which is one or the other
  //     when nvcc is able to deal with this, remove this WAR
  
  // given the minimal system, determine the unary transform_if functor we need
  typedef typename thrust::detail::unary_transform_if_functor<System,UnaryFunction,Predicate>::type UnaryTransformIfFunctor;

  // make an iterator tuple
  typedef thrust::tuple<InputIterator,ForwardIterator> IteratorTuple;
  typedef thrust::zip_iterator<IteratorTuple> ZipIterator;

  ZipIterator zipped_result =
    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(first,result)),
                     thrust::make_zip_iterator(thrust::make_tuple(last,result)),
                     UnaryTransformIfFunctor(unary_op,pred));

  return thrust::get<1>(zipped_result.get_iterator_tuple());
} // end transform_if()


template<typename InputIterator1,
         typename InputIterator2,
         typename ForwardIterator,
         typename UnaryFunction,
         typename Predicate>
  ForwardIterator transform_if(tag,
                               InputIterator1 first,
                               InputIterator1 last,
                               InputIterator2 stencil,
                               ForwardIterator result,
                               UnaryFunction unary_op,
                               Predicate pred)
{
  // determine the minimal system of the three iterators
  typedef typename thrust::iterator_system<InputIterator1>::type        System1;
  typedef typename thrust::iterator_system<InputIterator2>::type        System2;
  typedef typename thrust::iterator_system<ForwardIterator>::type       System3;

  typedef typename thrust::detail::minimum_system<System1,System2>::type  System4;
  typedef typename thrust::detail::minimum_system<System4,System3>::type  System;

  // XXX WAR the problem of a generic __host__ __device__ functor's inability to invoke
  //     a function which is only __host__ or __device__ by selecting a generic functor
  //     which is one or the other
  //     when nvcc is able to deal with this, remove this WAR
  
  // given the minimal system, determine the unary transform_if functor we need
  typedef typename thrust::detail::unary_transform_if_with_stencil_functor<System,UnaryFunction,Predicate>::type UnaryTransformIfFunctor;

  // make an iterator tuple
  typedef thrust::tuple<InputIterator1,InputIterator2,ForwardIterator> IteratorTuple;
  typedef thrust::zip_iterator<IteratorTuple> ZipIterator;

  ZipIterator zipped_result =
    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(first,stencil,result)),
                     thrust::make_zip_iterator(thrust::make_tuple(last,stencil,result)),
                     UnaryTransformIfFunctor(unary_op,pred));

  return thrust::get<2>(zipped_result.get_iterator_tuple());
} // end transform_if()


template<typename InputIterator1,
         typename InputIterator2,
         typename InputIterator3,
         typename ForwardIterator,
         typename BinaryFunction,
         typename Predicate>
  ForwardIterator transform_if(tag,
                               InputIterator1 first1,
                               InputIterator1 last1,
                               InputIterator2 first2,
                               InputIterator3 stencil,
                               ForwardIterator result,
                               BinaryFunction binary_op,
                               Predicate pred)
{
  // determine the minimal system of the four iterators
  typedef typename thrust::iterator_system<InputIterator1>::type        System1;
  typedef typename thrust::iterator_system<InputIterator2>::type        System2;
  typedef typename thrust::iterator_system<InputIterator3>::type        System3;
  typedef typename thrust::iterator_system<ForwardIterator>::type       System4;

  typedef typename thrust::detail::minimum_system<System1,System2>::type  System5;
  typedef typename thrust::detail::minimum_system<System3,System4>::type  System6;
  typedef typename thrust::detail::minimum_system<System5,System6>::type  System;

  // XXX WAR the problem of a generic __host__ __device__ functor's inability to invoke
  //     a function which is only __host__ or __device__ by selecting a generic functor
  //     which is one or the other
  //     when nvcc is able to deal with this, remove this WAR
  
  // given the minimal system, determine the binary transform_if functor we need
  typedef typename thrust::detail::binary_transform_if_functor<System,BinaryFunction,Predicate>::type BinaryTransformIfFunctor;

  // make an iterator tuple
  typedef thrust::tuple<InputIterator1,InputIterator2,InputIterator3,ForwardIterator> IteratorTuple;
  typedef thrust::zip_iterator<IteratorTuple> ZipIterator;

  ZipIterator zipped_result =
    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(first1,first2,stencil,result)),
                     thrust::make_zip_iterator(thrust::make_tuple(last1,first2,stencil,result)),
                     BinaryTransformIfFunctor(binary_op,pred));

  return thrust::get<3>(zipped_result.get_iterator_tuple());
} // end transform_if()


} // end generic
} // end detail
} // end system
} // end thrust

