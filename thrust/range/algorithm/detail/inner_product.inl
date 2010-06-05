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

#pragma once

#include <thrust/range/algorithm/inner_product.h>
#include <thrust/inner_product.h>
#include <thrust/range/begin.h>
#include <thrust/range/end.h>

namespace thrust
{

namespace experimental
{

namespace range
{


template<typename SinglePassRange1, typename SinglePassRange2, typename T>
  T inner_product(const SinglePassRange1 &rng1,
                  const SinglePassRange2 &rng2,
                  T init)
{
  return thrust::inner_product(begin(rng1), end(rng1), begin(rng2), init);
} // end inner_product()


template<typename SinglePassRange1, typename SinglePassRange2, typename T,
         typename BinaryOperation1, typename BinaryOperation2>
  T inner_product(const SinglePassRange1 &rng1,
                  const SinglePassRange2 &rng2,
                  T init,
                  BinaryOperation1 binary_op1,
                  BinaryOperation2 binary_op2)
{
  return thrust::inner_product(begin(rng1), end(rng1), begin(rng2), init, binary_op1, binary_op2);
} // end inner_product()


} // end range

} // end experimental

} // end thrust

