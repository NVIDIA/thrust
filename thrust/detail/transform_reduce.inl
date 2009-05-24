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


/*! \file reduce.inl
 *  \brief Inline file for transform_reduce.h.
 */

#include <thrust/transform_reduce.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/detail/dispatch/transform_reduce.h>

namespace thrust
{

template<typename InputIterator, 
         typename UnaryFunction, 
         typename OutputType,
         typename BinaryFunction>
  OutputType transform_reduce(InputIterator begin,
                              InputIterator end,
                              UnaryFunction unary_op,
                              OutputType init,
                              BinaryFunction binary_op)
{
  // dispatch on category
  return thrust::detail::dispatch::transform_reduce(begin, end, unary_op, init, binary_op,
    typename thrust::iterator_traits<InputIterator>::iterator_category());
} // end transform_reduce()

} // end namespace thrust

