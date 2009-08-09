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


/*! \file reduce.h
 *  \brief Device implementations for reduce.
 */

#pragma once

#include <thrust/iterator/iterator_traits.h>

#include <thrust/detail/type_traits.h>
#include <thrust/detail/device/cuda/reduce.h>
#include <thrust/detail/device/dereference.h>

namespace thrust
{

namespace detail
{

namespace device
{

namespace detail
{

//////////////    
// Functors //
//////////////    

template <typename InputIterator, typename OutputType>
struct reduce_functor
{
  InputIterator first;

  reduce_functor(InputIterator _first) 
    : first(_first)  {}

  template <typename IntegerType>
      __device__
      OutputType operator[](const IntegerType& i)
      { 
          return thrust::detail::device::dereference(first, i);
      }
}; // end reduce_functor

template <typename InputType, typename OutputType, typename BinaryFunction,
          typename WideType>
  struct wide_reduce_functor
{
  const InputType * input;
  BinaryFunction binary_op;

  __host__ __device__ 
  wide_reduce_functor(const InputType * _input, BinaryFunction _binary_op) 
    : input(_input), binary_op(_binary_op) {}

  template <typename IntegerType>
  __host__ __device__
  OutputType operator[](const IntegerType& i) {
      const WideType x = reinterpret_cast<const WideType *>(input)[i];
      WideType mask = ((WideType) 1 << (8 * sizeof(InputType))) - 1;
      OutputType sum = static_cast<InputType>(x & mask);
      for(unsigned int n = 1; n < sizeof(WideType) / sizeof(InputType); n++)
          sum = binary_op(sum, static_cast<InputType>( (x >> (8 * n * sizeof(InputType))) & mask ) );
      return sum;
  }
}; // end wide_reduce_functor


template<typename InputIterator, 
         typename OutputType,
         typename BinaryFunction>
  OutputType reduce_device(InputIterator first,
                           InputIterator last,
                           OutputType init,
                           BinaryFunction binary_op,
                           thrust::detail::true_type)
{
    // "wide" reduction for small types like char, short, etc.
    typedef typename thrust::iterator_traits<InputIterator>::value_type InputType;
    typedef unsigned int WideType;

    // note: this assumes that InputIterator is a InputType * and can be reinterpret_casted to WideType *
    
    // process first part
    size_t input_type_per_wide_type = sizeof(WideType) / sizeof(InputType);
    size_t n_wide = (last - first) / input_type_per_wide_type;
    wide_reduce_functor<InputType, OutputType, BinaryFunction, WideType> wide_func((&*first).get(), binary_op);
    OutputType result = thrust::detail::device::cuda::reduce(wide_func, n_wide, init, binary_op);

    // process tail
    InputIterator tail_first = first + n_wide * input_type_per_wide_type;
    reduce_functor<InputIterator, OutputType> tail_func(tail_first);
    return thrust::detail::device::cuda::reduce(tail_func, last - tail_first, result, binary_op);
}


template<typename InputIterator, 
         typename OutputType,
         typename BinaryFunction>
  OutputType reduce_device(InputIterator first,
                           InputIterator last,
                           OutputType init,
                           BinaryFunction binary_op,
                           thrust::detail::false_type)
{
    // standard reduction
    reduce_functor<InputIterator, OutputType> func(first);
    return thrust::detail::device::cuda::reduce(func, last - first, init, binary_op);
}

} // end namespace detail


template<typename InputIterator, 
         typename OutputType,
         typename BinaryFunction>
  OutputType reduce(InputIterator first,
                    InputIterator last,
                    OutputType init,
                    BinaryFunction binary_op)
{
    typedef typename thrust::iterator_traits<InputIterator>::value_type InputType;

    const bool use_wide_load = thrust::detail::is_pod<InputType>::value 
                                    && thrust::detail::is_trivial_iterator<InputIterator>::value
                                    && (sizeof(InputType) == 1 || sizeof(InputType) == 2);
                                    
    return detail::reduce_device(first, last, init, binary_op, thrust::detail::integral_constant<bool, use_wide_load>());
}

} // end namespace device

} // end namespace detail

} // end namespace thrust

