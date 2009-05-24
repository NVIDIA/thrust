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
 *  \brief Dispatch layer for inner_product.
 */

#pragma once

#include <thrust/iterator/iterator_categories.h>
#include <thrust/iterator/iterator_traits.h>

#include <numeric>
#include <thrust/detail/device/cuda/reduce.h>

namespace thrust
{

namespace detail
{

namespace dispatch
{

namespace detail
{

//////////////
// Functors //
//////////////
template <typename InputType1, typename InputType2, typename OutputType, typename BinaryFunction2>
struct inner_product_functor
{
  const InputType1 * input1;
  const InputType2 * input2;
  BinaryFunction2 binary_op2;

  __host__ __device__ 
  inner_product_functor(const InputType1 * _input1,
                        const InputType2 * _input2,
                        BinaryFunction2 _binary_op2) 
    : input1(_input1), input2(_input2), binary_op2(_binary_op2) {}

  template <typename IntegerType>
  __host__ __device__
  OutputType operator[](const IntegerType& i) { return binary_op2(input1[i], input2[i]); }
}; // end inner_product_functor

} // end detail

///////////////
// Host Path //
///////////////
template <typename InputIterator1, typename InputIterator2, typename OutputType,
          typename BinaryFunction1, typename BinaryFunction2>
    OutputType
    inner_product(InputIterator1 first1, InputIterator1 last1,
                  InputIterator2 first2, OutputType init, 
                  BinaryFunction1 binary_op1, BinaryFunction2 binary_op2,
                  thrust::input_host_iterator_tag,
                  thrust::input_host_iterator_tag)
{
    return std::inner_product(first1, last1, first2, init, binary_op1, binary_op2);
} 

/////////////////
// Device Path //
/////////////////
template <typename InputIterator1, typename InputIterator2, typename OutputType,
          typename BinaryFunction1, typename BinaryFunction2>
    OutputType
    inner_product(InputIterator1 first1, InputIterator1 last1,
                  InputIterator2 first2, OutputType init, 
                  BinaryFunction1 binary_op1, BinaryFunction2 binary_op2,
                  thrust::random_access_device_iterator_tag,
                  thrust::random_access_device_iterator_tag)
{
    typedef typename thrust::iterator_traits<InputIterator1>::value_type InputType1;
    typedef typename thrust::iterator_traits<InputIterator2>::value_type InputType2;

    // XXX use make_device_dereferenceable here instead of assuming &*first1 & &*first2 are device_ptr
    detail::inner_product_functor<InputType1, InputType2, OutputType, BinaryFunction2> 
        func((&*first1).get(), (&*first2).get(), binary_op2);

    return thrust::detail::device::cuda::reduce(func, last1 - first1, init, binary_op1);
}

} // end dispatch

} // end detail

} // end thrust

