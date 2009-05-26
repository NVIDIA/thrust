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


/*! \file transform.h
 *  \brief Defines the interface
 *         to the dispatch layer of
 *         the family of transform functions.
 */

#pragma once

#include <algorithm>
#include <thrust/iterator/iterator_categories.h>
#include <thrust/iterator/iterator_traits.h>

#include <thrust/detail/host/transform.h>
#include <thrust/detail/device/cuda/vectorize.h>

namespace thrust
{

namespace detail
{

namespace dispatch
{

namespace detail
{

template <typename InputType, typename OutputType, typename UnaryFunction>
struct unary_transform_functor
{
  const InputType  * input;
        OutputType * output;
  UnaryFunction unary_op;

  unary_transform_functor(const InputType * _input, OutputType * _output, UnaryFunction _unary_op)
    : input(_input), output(_output), unary_op(_unary_op) {} 
  
  template <typename IntegerType>
  __host__ __device__
  void operator()(const IntegerType& i) { output[i] = unary_op(input[i]); }
}; // end unary_transform_functor


template <typename InputType1, typename InputType2, typename OutputType, typename BinaryFunction>
struct binary_transform_functor
{
  const InputType1 * input1;
  const InputType2 * input2;
        OutputType * output;
  BinaryFunction binary_op;

  binary_transform_functor(const InputType1 * _input1, const InputType2 * _input2, OutputType * _output, BinaryFunction _binary_op)
    : input1(_input1), input2(_input2), output(_output), binary_op(_binary_op) {} 
  
  template <typename IntegerType>
  __host__ __device__
  void operator()(const IntegerType& i) { output[i] = binary_op(input1[i], input2[i]); }
}; // end binary_transform_functor

} // end detail

template<typename InputIterator,
         typename OutputIterator,
         typename UnaryFunction>
  OutputIterator transform(InputIterator first, InputIterator last,
                           OutputIterator result,
                           UnaryFunction op,
                           thrust::input_host_iterator_tag,
                           thrust::output_host_iterator_tag)
{
  return std::transform(first, last, result, op);
} // end transform()


template<typename InputIterator,
         typename OutputIterator,
         typename UnaryFunction>
  OutputIterator transform(InputIterator first, InputIterator last,
                           OutputIterator result,
                           UnaryFunction op,
                           thrust::input_host_iterator_tag,
                           thrust::input_host_iterator_tag)
{
  return std::transform(first, last, result, op);
} // end transform()


template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename BinaryFunction>
  OutputIterator transform(InputIterator1 first1, InputIterator1 last1,
                           InputIterator2 first2,
                           OutputIterator result,
                           BinaryFunction op,
                           thrust::input_host_iterator_tag,
                           thrust::input_host_iterator_tag,
                           thrust::output_host_iterator_tag)
{
  return std::transform(first1, last1, first2, result, op);
} // end transform()


template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename BinaryFunction>
  OutputIterator transform(InputIterator1 first1, InputIterator1 last1,
                           InputIterator2 first2,
                           OutputIterator result,
                           BinaryFunction op,
                           thrust::input_host_iterator_tag,
                           thrust::input_host_iterator_tag,
                           thrust::input_host_iterator_tag)
{
  return std::transform(first1, last1, first2, result, op);
} // end transform()


template<typename InputIterator,
         typename OutputIterator,
         typename UnaryFunction>
  OutputIterator transform(InputIterator first, InputIterator last,
                           OutputIterator result,
                           UnaryFunction op,
                           thrust::random_access_device_iterator_tag,
                           thrust::random_access_device_iterator_tag)
{
  typedef typename thrust::iterator_traits<InputIterator>::value_type InputType;
  typedef typename thrust::iterator_traits<OutputIterator>::value_type OutputType;

  // XXX use make_device_dereferenceable here instead of assuming &*first & &*result are device_ptr
  detail::unary_transform_functor<InputType,OutputType,UnaryFunction> func((&*first).get(), (&*result).get(), op);
  thrust::detail::device::cuda::vectorize(last - first, func);

  return result + (last - first); // return the end of the output sequence
} // end transform() 


template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename BinaryFunction>
  OutputIterator transform(InputIterator1 first1, InputIterator1 last1,
                           InputIterator2 first2,
                           OutputIterator result,
                           BinaryFunction op,
                           thrust::random_access_device_iterator_tag,
                           thrust::random_access_device_iterator_tag,
                           thrust::random_access_device_iterator_tag)
{
  typedef typename thrust::iterator_traits<InputIterator1>::value_type InputType1;
  typedef typename thrust::iterator_traits<InputIterator2>::value_type InputType2;
  typedef typename thrust::iterator_traits<OutputIterator>::value_type OutputType;

  // XXX use make_device_dereferenceable here instead of assuming &*first1, &*first2, &*result are device_ptr
  detail::binary_transform_functor<InputType1,InputType2,OutputType,BinaryFunction> func((&*first1).get(), (&*first2).get(), (&*result).get(), op);
  thrust::detail::device::cuda::vectorize(last1 - first1, func);

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
                               Predicate pred,
                               thrust::input_host_iterator_tag,
                               thrust::input_host_iterator_tag,
                               thrust::forward_host_iterator_tag)
{
  return thrust::detail::host::transform_if(first, last, stencil, result, unary_op, pred);
} // end transform_if()


template<typename InputIterator1,
         typename InputIterator2,
         typename ForwardIterator,
         typename UnaryFunction,
         typename Predicate>
  ForwardIterator transform_if(InputIterator1 first, InputIterator1 last,
                               InputIterator2 stencil,
                               ForwardIterator result,
                               UnaryFunction unary_op,
                               Predicate pred,
                               thrust::input_host_iterator_tag,
                               thrust::input_host_iterator_tag,
                               thrust::input_host_iterator_tag)
{
  return thrust::detail::host::transform_if(first, last, stencil, result, unary_op, pred);
} // end transform_if()


template<typename InputIterator1,
         typename InputIterator2,
         typename InputIterator3,
         typename ForwardIterator,
         typename UnaryFunction,
         typename Predicate>
  ForwardIterator transform_if(InputIterator1 first1, InputIterator1 last1,
                               InputIterator2 first2,
                               InputIterator3 stencil,
                               ForwardIterator result,
                               UnaryFunction unary_op,
                               Predicate pred,
                               thrust::input_host_iterator_tag,
                               thrust::input_host_iterator_tag,
                               thrust::input_host_iterator_tag,
                               thrust::forward_host_iterator_tag)
{
  return thrust::detail::host::transform_if(first1, last1, first2, stencil, result, unary_op, pred);
} // end transform_if()


template<typename InputIterator1,
         typename InputIterator2,
         typename InputIterator3,
         typename ForwardIterator,
         typename UnaryFunction,
         typename Predicate>
  ForwardIterator transform_if(InputIterator1 first1, InputIterator1 last1,
                               InputIterator2 first2,
                               InputIterator3 stencil,
                               ForwardIterator result,
                               UnaryFunction unary_op,
                               Predicate pred,
                               thrust::input_host_iterator_tag,
                               thrust::input_host_iterator_tag,
                               thrust::input_host_iterator_tag,
                               thrust::input_host_iterator_tag)
{
  return thrust::detail::host::transform_if(first1, last1, first2, stencil, result, unary_op, pred);
} // end transform_if()


namespace detail
{

template <typename InputType1, typename InputType2, typename OutputType, typename UnaryFunction, typename Predicate>
struct unary_transform_if_functor
{
  const InputType1 * input;
  const InputType2 * stencil;
        OutputType * output;
  UnaryFunction unary_op;
  Predicate pred;

  unary_transform_if_functor(const InputType1 * _input,
                             const InputType2 * _stencil,
                             OutputType * _output,
                             UnaryFunction _unary_op,
                             Predicate _pred)
    : input(_input), stencil(_stencil), output(_output), unary_op(_unary_op), pred(_pred) {} 
  
  template <typename IntegerType>
  __host__ __device__
  void operator()(const IntegerType& i)
  {
    if(pred(stencil[i]))
      output[i] = unary_op(input[i]);
  }
}; // end unary_transform_if_functor


template <typename InputType1, typename InputType2, typename InputType3, typename OutputType, typename BinaryFunction, typename Predicate>
struct binary_transform_if_functor
{
  const InputType1 * input1;
  const InputType2 * input2;
  const InputType3 * stencil;
        OutputType * output;
  BinaryFunction binary_op;
  Predicate pred;

  binary_transform_if_functor(const InputType1 * _input1,
                              const InputType2 * _input2,
                              const InputType3 * _stencil,
                              OutputType * _output,
                              BinaryFunction _binary_op,
                              Predicate _pred)
    : input1(_input1), input2(_input2), stencil(_stencil), output(_output), binary_op(_binary_op), pred(_pred) {} 
  
  template <typename IntegerType>
  __host__ __device__
  void operator()(const IntegerType& i)
  {
    if(pred(stencil[i]))
      output[i] = binary_op(input1[i], input2[i]);
  }
}; // end binary_transform_if_functor

} // end detail


template<typename InputIterator1,
         typename InputIterator2,
         typename ForwardIterator,
         typename UnaryFunction,
         typename Predicate>
  ForwardIterator transform_if(InputIterator1 first, InputIterator1 last,
                               InputIterator2 stencil,
                               ForwardIterator result,
                               UnaryFunction unary_op,
                               Predicate pred,
                               thrust::random_access_device_iterator_tag,
                               thrust::random_access_device_iterator_tag,
                               thrust::random_access_device_iterator_tag)
{
  typedef typename thrust::iterator_traits<InputIterator1>::value_type InputType1;
  typedef typename thrust::iterator_traits<InputIterator2>::value_type InputType2;
  typedef typename thrust::iterator_traits<ForwardIterator>::value_type OutputType;

  typedef detail::unary_transform_if_functor<InputType1,InputType2,OutputType,UnaryFunction,Predicate> Functor;

  // XXX use make_device_dereferenceable here instead of assuming &*first & &*result are device_ptr
  Functor func((&*first).get(), (&*stencil).get(), (&*result).get(), unary_op, pred);
  thrust::detail::device::cuda::vectorize(last - first, func);

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
                               Predicate pred,
                               thrust::random_access_device_iterator_tag,
                               thrust::random_access_device_iterator_tag,
                               thrust::random_access_device_iterator_tag,
                               thrust::random_access_device_iterator_tag)
{
  typedef typename thrust::iterator_traits<InputIterator1>::value_type InputType1;
  typedef typename thrust::iterator_traits<InputIterator2>::value_type InputType2;
  typedef typename thrust::iterator_traits<InputIterator3>::value_type InputType3;
  typedef typename thrust::iterator_traits<ForwardIterator>::value_type OutputType;

  typedef detail::binary_transform_if_functor<InputType1,InputType2,InputType3,OutputType,BinaryFunction,Predicate> Functor;

  // XXX use make_device_dereferenceable here instead of assuming &*first & &*result are device_ptr
  Functor func((&*first1).get(), (&*first2).get(), (&*stencil).get(), (&*result).get(), binary_op, pred);
  thrust::detail::device::cuda::vectorize(last1 - first1, func);

  return result + (last1 - first1); // return the end of the output sequence
} // end transform_if()

} // end dispatch

} // end detail

} // end thrust

