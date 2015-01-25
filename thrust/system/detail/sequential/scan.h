/*
 *  Copyright 2008-2013 NVIDIA Corporation
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


/*! \file scan.h
 *  \brief Sequential implementations of scan functions.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/system/detail/sequential/execution_policy.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/detail/type_traits.h>
#include <thrust/detail/type_traits/function_traits.h>
#include <thrust/detail/type_traits/iterator/is_output_iterator.h>
#include <thrust/detail/function.h>

namespace thrust
{
namespace system
{
namespace detail
{
namespace sequential
{


__thrust_hd_warning_disable__
template<typename DerivedPolicy,
         typename InputIterator,
         typename OutputIterator,
         typename BinaryFunction>
__host__ __device__
  OutputIterator inclusive_scan(sequential::execution_policy<DerivedPolicy> &,
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
  
  using namespace thrust::detail;

  typedef typename eval_if<
    has_result_type<BinaryFunction>::value,
    result_type<BinaryFunction>,
    eval_if<
      is_output_iterator<OutputIterator>::value,
      thrust::iterator_value<InputIterator>,
      thrust::iterator_value<OutputIterator>
    >
  >::type ValueType;

  // wrap binary_op
  thrust::detail::wrapped_function<
    BinaryFunction,
    ValueType
  > wrapped_binary_op(binary_op);

  if(first != last)
  {
    ValueType sum = *first;

    *result = sum;

    for(++first, ++result; first != last; ++first, ++result)
      *result = sum = wrapped_binary_op(sum,*first);
  }

  return result;
}


__thrust_hd_warning_disable__
template<typename DerivedPolicy,
         typename InputIterator,
         typename OutputIterator,
         typename T,
         typename BinaryFunction>
__host__ __device__
  OutputIterator exclusive_scan(sequential::execution_policy<DerivedPolicy> &,
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

  using namespace thrust::detail;

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
    ValueType tmp = *first;  // temporary value allows in-situ scan
    ValueType sum = init;

    *result = sum;
    sum = binary_op(sum, tmp);

    for(++first, ++result; first != last; ++first, ++result)
    {
      tmp = *first;
      *result = sum;
      sum = binary_op(sum, tmp);
    }
  }

  return result;
} 


} // end namespace sequential
} // end namespace detail
} // end namespace system
} // end namespace thrust

