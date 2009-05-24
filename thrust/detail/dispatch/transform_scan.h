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


/*! \file transform_scan.h
 *  \brief Defines the interface to the
 *         dispatch layer of the family of
 *         transform scan functions.
 */

#pragma once

#include <thrust/functional.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/iterator_categories.h>

#include <thrust/detail/host/transform_scan.h>
#include <thrust/detail/device/cuda/scan.h>

namespace thrust
{

namespace detail
{

namespace dispatch
{

namespace detail
{

template <typename InputType, typename OutputType, typename UnaryFunction>
struct transform_scan_functor
{
  const InputType * input;
  UnaryFunction unary_op;

  __host__ __device__ 
  transform_scan_functor(const InputType * _input, UnaryFunction _unary_op) 
    : input(_input), unary_op(_unary_op) {}

  template <typename IntegerType>
  __host__ __device__
  OutputType operator[](const IntegerType& i) { return unary_op(input[i]); }
}; // end transform_scan_functor

} // end namespace detail


////////////////
// Host Paths //
////////////////

template<typename InputIterator,
         typename OutputIterator,
         typename UnaryFunction,
         typename AssociativeOperator>
  void transform_inclusive_scan(InputIterator begin,
                                InputIterator end,
                                OutputIterator result,
                                UnaryFunction unary_op,
                                AssociativeOperator binary_op,
                                thrust::forward_host_iterator_tag,
                                thrust::forward_host_iterator_tag)
{
    thrust::detail::host::transform_inclusive_scan(begin, end, result, unary_op, binary_op);
}


template<typename InputIterator,
         typename OutputIterator,
         typename UnaryFunction,
         typename T,
         typename AssociativeOperator>
  void transform_exclusive_scan(InputIterator begin,
                                InputIterator end,
                                OutputIterator result,
                                UnaryFunction unary_op,
                                T init,
                                AssociativeOperator binary_op,
                                thrust::forward_host_iterator_tag,
                                thrust::forward_host_iterator_tag)
{
    thrust::detail::host::transform_exclusive_scan(begin, end, result, unary_op, init, binary_op);
}


//////////////////
// Device Paths //
//////////////////

template<typename InputIterator,
         typename OutputIterator,
         typename UnaryFunction,
         typename AssociativeOperator>
  void transform_inclusive_scan(InputIterator begin,
                                InputIterator end,
                                OutputIterator result,
                                UnaryFunction unary_op,
                                AssociativeOperator binary_op,
                                thrust::random_access_device_iterator_tag,
                                thrust::random_access_device_iterator_tag)
{
  size_t n = end - begin;
  typedef typename thrust::iterator_traits<InputIterator>::value_type InputType;
  typedef typename thrust::iterator_traits<OutputIterator>::value_type OutputType;

  // XXX use make_device_dereferenceable here instead of assuming &*begin is device_ptr
  detail::transform_scan_functor<InputType,OutputType,UnaryFunction> func((&*begin).get(), unary_op);

  // XXX use make_device_dereferenceable here instead of assuming &*result is device_ptr
  return thrust::detail::device::cuda::inclusive_scan(func, n, (&*result).get(), binary_op);
}


template<typename InputIterator,
         typename OutputIterator,
         typename UnaryFunction,
         typename T,
         typename AssociativeOperator>
  void transform_exclusive_scan(InputIterator begin,
                                InputIterator end,
                                OutputIterator result,
                                UnaryFunction unary_op,
                                T init,
                                AssociativeOperator binary_op,
                                thrust::random_access_device_iterator_tag,
                                thrust::random_access_device_iterator_tag)
{
  size_t n = end - begin;
  typedef typename thrust::iterator_traits<InputIterator>::value_type InputType;
  typedef typename thrust::iterator_traits<OutputIterator>::value_type OutputType;

  // XXX use make_device_dereferenceable here instead of assuming &*begin is device_ptr
  detail::transform_scan_functor<InputType,OutputType,UnaryFunction> func((&*begin).get(), unary_op);

  // XXX use make_device_dereferenceable here instead of assuming &*result is device_ptr
  return thrust::detail::device::cuda::exclusive_scan(func, n, (&*result).get(), OutputType(init), binary_op);
}

} // end namespace dispatch

} // end namespace detail

} // end namespace thrust

