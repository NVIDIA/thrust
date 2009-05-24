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


/*! \file transform_reduce.h
 *  \brief Dispatch layer for transform_reduce.
 */

#pragma once

#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/iterator_categories.h>

#include <thrust/detail/host/transform_reduce.h>
#include <thrust/detail/device/cuda/reduce.h>
#include <thrust/detail/type_traits.h>

namespace thrust
{

namespace detail
{

namespace dispatch
{

namespace detail
{

template <typename InputType, typename UnaryFunction, typename OutputType>
  struct transform_reduce_functor
{
  const InputType * input;
  UnaryFunction unary_op;

  __host__ __device__ 
  transform_reduce_functor(const InputType * _input, UnaryFunction _unary_op) 
    : input(_input), unary_op(_unary_op) {}

  template <typename IntegerType>
  __host__ __device__
  OutputType operator[](const IntegerType& i) { return unary_op(input[i]); }
}; // end transform_reduce_functor


template <typename InputType, typename UnaryFunction, typename OutputType, typename BinaryFunction,
          typename WideType>
  struct wide_transform_reduce_functor
{
  const InputType * input;
  UnaryFunction unary_op;
  BinaryFunction binary_op;

  __host__ __device__ 
  wide_transform_reduce_functor(const InputType * _input, UnaryFunction _unary_op, BinaryFunction _binary_op) 
    : input(_input), unary_op(_unary_op), binary_op(_binary_op) {}

  template <typename IntegerType>
  __host__ __device__
  OutputType operator[](const IntegerType& i) {
      const WideType x = reinterpret_cast<const WideType *>(input)[i];
      WideType mask = ((WideType) 1 << (8 * sizeof(InputType))) - 1;
      OutputType sum = unary_op(static_cast<InputType>(x & mask));
      for(unsigned int n = 1; n < sizeof(WideType) / sizeof(InputType); n++)
          sum = binary_op(sum, unary_op( static_cast<InputType>( (x >> (8 * n * sizeof(InputType))) & mask ) ));
      return sum;
  }
}; // end wide_transform_reduce_functor


template<typename InputIterator, 
         typename UnaryFunction, 
         typename OutputType,
         typename BinaryFunction>
  OutputType transform_reduce_device(InputIterator begin,
                                     InputIterator end,
                                     UnaryFunction unary_op,
                                     OutputType init,
                                     BinaryFunction binary_op,
                                     thrust::detail::util::Bool2Type<true>)
{
  typedef typename thrust::iterator_traits<InputIterator>::value_type InputType;
  typedef unsigned int WideType;
    
  // XXX use make_device_dereferenceable here instead of assuming &*begin is device_ptr
  wide_transform_reduce_functor<InputType, UnaryFunction, OutputType, BinaryFunction, WideType> wide_func((&*begin).get(), unary_op, binary_op);

  size_t input_type_per_wide_type = sizeof(WideType) / sizeof(InputType);
  size_t n_wide = (end - begin) / input_type_per_wide_type;

  OutputType result = thrust::detail::device::cuda::reduce(wide_func, n_wide, init, binary_op);

  InputIterator tail_begin = begin + n_wide * input_type_per_wide_type;

  // XXX use make_device_dereferenceable here instead of assuming &*begin is device_ptr
  detail::transform_reduce_functor<InputType,UnaryFunction,OutputType> func((&*tail_begin).get(), unary_op);
  
  return thrust::detail::device::cuda::reduce(func, end - tail_begin, result, binary_op);
}


template<typename InputIterator, 
         typename OutputType,
         typename UnaryFunction, 
         typename BinaryFunction>
  OutputType transform_reduce_device(InputIterator begin,
                                     InputIterator end,
                                     UnaryFunction unary_op,
                                     OutputType init,
                                     BinaryFunction binary_op,
                                     thrust::detail::util::Bool2Type<false>)
{
  typedef typename thrust::iterator_traits<InputIterator>::value_type InputType;
  
  // XXX use make_device_dereferenceable here instead of assuming &*begin is device_ptr
  detail::transform_reduce_functor<InputType,UnaryFunction,OutputType> func((&*begin).get(), unary_op);

  return thrust::detail::device::cuda::reduce(func, end - begin, init, binary_op);
}


} // end detail


///////////////
// Host Path //
///////////////

template<typename InputIterator, 
         typename UnaryFunction, 
         typename OutputType,
         typename BinaryFunction>
  OutputType transform_reduce(InputIterator begin,
                              InputIterator end,
                              UnaryFunction unary_op,
                              OutputType init,
                              BinaryFunction binary_op,
                              thrust::input_host_iterator_tag)
{
    return thrust::detail::host::transform_reduce(begin, end, unary_op, init, binary_op);
}


/////////////////
// Device Path //
/////////////////

template<typename InputIterator, 
         typename UnaryFunction, 
         typename OutputType,
         typename BinaryFunction>
  OutputType transform_reduce(InputIterator begin,
                              InputIterator end,
                              UnaryFunction unary_op,
                              OutputType init,
                              BinaryFunction binary_op,
                              thrust::random_access_device_iterator_tag)
{
    typedef typename thrust::iterator_traits<InputIterator>::value_type InputType;
    const bool use_wide_load = thrust::detail::is_pod<InputType>::value && (sizeof(InputType) == 1 || sizeof(InputType) == 2);
    return detail::transform_reduce_device(begin, end, unary_op, init, binary_op, thrust::detail::util::Bool2Type<use_wide_load>());
}

} // end dispatch

} // end detail

} // end thrust

