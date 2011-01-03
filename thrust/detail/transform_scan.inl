/*
 *  Copyright 2008-2011 NVIDIA Corporation
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

#include <thrust/scan.h>

#include <thrust/iterator/transform_iterator.h>

namespace thrust
{

template<typename InputIterator,
         typename OutputIterator,
         typename UnaryFunction,
         typename AssociativeOperator>
  OutputIterator transform_inclusive_scan(InputIterator first,
                                          InputIterator last,
                                          OutputIterator result,
                                          UnaryFunction unary_op,
                                          AssociativeOperator binary_op)
{
    typedef typename thrust::iterator_traits<OutputIterator>::value_type OutputType;

    thrust::transform_iterator<UnaryFunction, InputIterator, OutputType> _first(first, unary_op);
    thrust::transform_iterator<UnaryFunction, InputIterator, OutputType> _last(last, unary_op);

    return thrust::inclusive_scan(_first, _last, result, binary_op);
}


template<typename InputIterator,
         typename OutputIterator,
         typename UnaryFunction,
         typename T,
         typename AssociativeOperator>
  OutputIterator transform_exclusive_scan(InputIterator first,
                                          InputIterator last,
                                          OutputIterator result,
                                          UnaryFunction unary_op,
                                          T init,
                                          AssociativeOperator binary_op)
{
    typedef typename thrust::iterator_traits<OutputIterator>::value_type OutputType;
    
    thrust::transform_iterator<UnaryFunction, InputIterator, OutputType> _first(first, unary_op);
    thrust::transform_iterator<UnaryFunction, InputIterator, OutputType> _last(last, unary_op);

    return thrust::exclusive_scan(_first, _last, result, init, binary_op);
}

} // end namespace thrust

