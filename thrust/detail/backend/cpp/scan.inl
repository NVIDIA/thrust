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

#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/backend/cpp/scan.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/detail/type_traits.h>
#include <thrust/detail/type_traits/function_traits.h>
#include <thrust/detail/type_traits/iterator/is_output_iterator.h>
#include <thrust/detail/backend/dereference.h>

namespace thrust
{
namespace detail
{
namespace backend
{
namespace cpp
{

template<typename InputIterator,
         typename OutputIterator,
         typename BinaryFunction>
  OutputIterator inclusive_scan(tag,
                                InputIterator first,
                                InputIterator last,
                                OutputIterator result,
                                BinaryFunction binary_op)
{
  // the pseudocode for deducing the type of the temporary used below:
  // 
  // if BinaryFunction is AdaptableBinaryFunction
  //   TemporaryType = AdaptableBinaryFunction::result_type
  // else if OutputIterator is a "pure" output iterator
  //   TemporaryType = InputIterator::value_type
  // else
  //   TemporaryType = OutputIterator::value_type
  //
  // XXX upon c++0x, TemporaryType needs to be:
  // result_of<BinaryFunction>::type

  typedef typename eval_if<
    has_result_type<BinaryFunction>::value,
    result_type<BinaryFunction>,
    eval_if<
      is_output_iterator<OutputIterator>::value,
      thrust::iterator_value<InputIterator>,
      thrust::iterator_value<OutputIterator>
    >
  >::type ValueType;

  if(first != last)
  {
    ValueType sum = backend::dereference(first);

    backend::dereference(result) = sum;

    for(++first, ++result; first != last; ++first, ++result)
      backend::dereference(result) = sum = binary_op(sum, backend::dereference(first));
  }

  return result;
}


template<typename InputIterator,
         typename OutputIterator,
         typename T,
         typename BinaryFunction>
  OutputIterator exclusive_scan(tag,
                                InputIterator first,
                                InputIterator last,
                                OutputIterator result,
                                T init,
                                BinaryFunction binary_op)
{
  // the pseudocode for deducing the type of the temporary used below:
  // 
  // if BinaryFunction is AdaptableBinaryFunction
  //   TemporaryType = AdaptableBinaryFunction::result_type
  // else if OutputIterator is a "pure" output iterator
  //   TemporaryType = InputIterator::value_type
  // else
  //   TemporaryType = OutputIterator::value_type
  //
  // XXX upon c++0x, TemporaryType needs to be:
  // result_of<BinaryFunction>::type

  typedef typename eval_if<
    has_result_type<BinaryFunction>::value,
    result_type<BinaryFunction>,
    eval_if<
      is_output_iterator<OutputIterator>::value,
      thrust::iterator_value<InputIterator>,
      thrust::iterator_value<OutputIterator>
    >
  >::type ValueType;

  if(first != last)
  {
    ValueType tmp = backend::dereference(first);  // temporary value allows in-situ scan
    ValueType sum = init;

    backend::dereference(result) = sum;
    sum = binary_op(sum, tmp);

    for(++first, ++result; first != last; ++first, ++result)
    {
      tmp = backend::dereference(first);
      backend::dereference(result) = sum;
      sum = binary_op(sum, tmp);
    }
  }

  return result;
} 


} // end namespace cpp
} // end namespace backend
} // end namespace detail
} // end namespace thrust

