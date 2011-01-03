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


/*! \file adjacent_difference.inl
 *  \brief Inline file for adjacent_difference.h
 */

#include <thrust/functional.h>

#include <thrust/detail/dispatch/adjacent_difference.h>

namespace thrust
{

template <class InputIterator, class OutputIterator>
OutputIterator adjacent_difference(InputIterator first, InputIterator last, 
                                   OutputIterator result)
{
  typedef typename thrust::iterator_traits<InputIterator>::value_type InputType;

  thrust::minus<InputType> binary_op;
  return thrust::adjacent_difference(first, last, result, binary_op);
} // end adjacent_difference()

template <class InputIterator, class OutputIterator, class BinaryFunction>
OutputIterator adjacent_difference(InputIterator first, InputIterator last,
                                   OutputIterator result,
                                   BinaryFunction binary_op)
{
  // dispatch on space
  return thrust::detail::dispatch::adjacent_difference(first, last, result, binary_op,
    typename thrust::iterator_space<InputIterator>::type(),
    typename thrust::iterator_space<OutputIterator>::type());
} // end adjacent_difference()


} // end namespace thrust

