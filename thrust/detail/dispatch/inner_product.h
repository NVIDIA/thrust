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


/*! \file inner_product.h
 *  \brief Dispatch layer for inner_product.
 */

#pragma once

#include <thrust/iterator/iterator_traits.h>

#include <numeric>
#include <thrust/detail/device/inner_product.h>

namespace thrust
{

namespace detail
{

namespace dispatch
{

///////////////
// Host Path //
///////////////
template <typename InputIterator1, typename InputIterator2, typename OutputType,
          typename BinaryFunction1, typename BinaryFunction2>
    OutputType
    inner_product(InputIterator1 first1, InputIterator1 last1,
                  InputIterator2 first2, OutputType init, 
                  BinaryFunction1 binary_op1, BinaryFunction2 binary_op2,
                  thrust::host_space_tag,
                  thrust::host_space_tag)
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
                  thrust::device_space_tag,
                  thrust::device_space_tag)
{
    return thrust::detail::device::inner_product(first1, last1, first2, init, binary_op1, binary_op2);    
}

} // end dispatch

} // end detail

} // end thrust

