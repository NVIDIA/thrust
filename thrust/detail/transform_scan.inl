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


/*! \file transform_scan.inl
 *  \brief Inline file for transform_scan.h.
 */

#include <thrust/transform_scan.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/detail/dispatch/transform_scan.h>

namespace thrust
{

template<typename InputIterator,
         typename OutputIterator,
         typename UnaryFunction,
         typename AssociativeOperator>
  void transform_inclusive_scan(InputIterator first,
                                InputIterator last,
                                OutputIterator result,
                                UnaryFunction unary_op,
                                AssociativeOperator binary_op)
{
    // dispatch on iterator category
    thrust::detail::dispatch::transform_inclusive_scan(first, last, result, unary_op, binary_op,
            typename thrust::iterator_traits<InputIterator>::iterator_category(),
            typename thrust::iterator_traits<OutputIterator>::iterator_category());
}


template<typename InputIterator,
         typename OutputIterator,
         typename UnaryFunction,
         typename T,
         typename AssociativeOperator>
  void transform_exclusive_scan(InputIterator first,
                                InputIterator last,
                                OutputIterator result,
                                UnaryFunction unary_op,
                                T init,
                                AssociativeOperator binary_op)
{
    // dispatch on category
    thrust::detail::dispatch::transform_exclusive_scan(first, last, result, unary_op, init, binary_op,
            typename thrust::iterator_traits<InputIterator>::iterator_category(),
            typename thrust::iterator_traits<OutputIterator>::iterator_category());
}

} // end namespace thrust

