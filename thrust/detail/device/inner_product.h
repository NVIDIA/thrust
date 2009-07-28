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


/*! \file inner_product.h
 *  \brief Device implementations for inner_product.
 */

#pragma once

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

template <typename InputIterator1, typename InputIterator2, typename OutputType, typename BinaryFunction2>
struct inner_product_functor
{
  InputIterator1 first1;
  InputIterator2 first2;
  BinaryFunction2 binary_op2;

  inner_product_functor(InputIterator1 _first1, InputIterator2 _first2, BinaryFunction2 _binary_op2) 
    : first1(_first1), first2(_first2), binary_op2(_binary_op2) {}

  template <typename IntegerType>
      __device__
      OutputType operator[](const IntegerType& i)
      { 
          return binary_op2(thrust::detail::device::dereference(first1, i), thrust::detail::device::dereference(first2, i));
      }
}; // end inner_product_functor

} // end namespace detail

template <typename InputIterator1, typename InputIterator2, typename OutputType,
          typename BinaryFunction1, typename BinaryFunction2>
    OutputType
    inner_product(InputIterator1 first1, InputIterator1 last1,
                  InputIterator2 first2, OutputType init, 
                  BinaryFunction1 binary_op1, BinaryFunction2 binary_op2)
{
    detail::inner_product_functor<InputIterator1, InputIterator2, OutputType, BinaryFunction2> 
        func(first1, first2, binary_op2);

    return thrust::detail::device::cuda::reduce(func, last1 - first1, init, binary_op1);
}

} // end namespace device

} // end namespace detail

} // end namespace thrust


