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
#include <thrust/system/detail/generic/transform_reduce.h>
#include <thrust/reduce.h>
#include <thrust/iterator/transform_iterator.h>

namespace thrust
{
namespace system
{
namespace detail
{
namespace generic
{

template<typename System,
         typename InputIterator, 
         typename UnaryFunction, 
         typename OutputType,
         typename BinaryFunction>
  OutputType transform_reduce(thrust::dispatchable<System> &system,
                              InputIterator first,
                              InputIterator last,
                              UnaryFunction unary_op,
                              OutputType init,
                              BinaryFunction binary_op)
{
  thrust::transform_iterator<UnaryFunction, InputIterator, OutputType> xfrm_first(first, unary_op);
  thrust::transform_iterator<UnaryFunction, InputIterator, OutputType> xfrm_last(last, unary_op);

  return thrust::reduce(system, xfrm_first, xfrm_last, init, binary_op);
} // end transform_reduce()

} // end generic
} // end detail
} // end system
} // end thrust

