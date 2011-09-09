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


/*! \file scan.h
 *  \brief C++ implementations of scan functions.
 */

#pragma once

#include <thrust/iterator/iterator_traits.h>

#include <thrust/detail/type_traits.h>
#include <thrust/detail/type_traits/function_traits.h>
#include <thrust/detail/type_traits/iterator/is_output_iterator.h>

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
  OutputIterator inclusive_scan(InputIterator first,
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
    ValueType sum = *first;

    *result = sum;

    for(++first, ++result; first != last; ++first, ++result)
      *result = sum = binary_op(sum, *first);
  }

  return result;
}


template<typename InputIterator,
         typename OutputIterator,
         typename T,
         typename BinaryFunction>
  OutputIterator exclusive_scan(InputIterator first,
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
    ValueType tmp = *first;  // temporary value allows in-situ scan
    ValueType sum =  init;

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


template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename BinaryPredicate,
         typename AssociativeOperator>
  OutputIterator inclusive_scan_by_key(InputIterator1 first1,
                                       InputIterator1 last1,
                                       InputIterator2 first2,
                                       OutputIterator result,
                                       BinaryPredicate binary_pred,
                                       AssociativeOperator binary_op)
{
    typedef typename thrust::iterator_traits<InputIterator1>::value_type KeyType;
    typedef typename thrust::iterator_traits<OutputIterator>::value_type ValueType;

    if(first1 != last1)
    {
        KeyType   prev_key   = *first1;
        ValueType prev_value = *first2;

        *result = prev_value;

        for(++first1, ++first2, ++result;
                first1 != last1;
                ++first1, ++first2, ++result)
        {
            KeyType key = *first1;

            if (binary_pred(prev_key, key))
                *result = prev_value = binary_op(prev_value, *first2);
            else
                *result = prev_value = *first2;

            prev_key = key;
        }
    }

    return result;
}


template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename T,
         typename BinaryPredicate,
         typename AssociativeOperator>
  OutputIterator exclusive_scan_by_key(InputIterator1 first1,
                                       InputIterator1 last1,
                                       InputIterator2 first2,
                                       OutputIterator result,
                                       T init,
                                       BinaryPredicate binary_pred,
                                       AssociativeOperator binary_op)
{
    typedef typename thrust::iterator_traits<InputIterator1>::value_type KeyType;
    typedef typename thrust::iterator_traits<OutputIterator>::value_type ValueType;

    if(first1 != last1)
    {
        KeyType   temp_key   = *first1;
        ValueType temp_value = *first2;
        
        ValueType next = init;

        // first one is init
        *result = next;

        next = binary_op(next, temp_value);

        for(++first1, ++first2, ++result;
                first1 != last1;
                ++first1, ++first2, ++result)
        {
            KeyType key = *first1;

            // use temp to permit in-place scans
            temp_value = *first2;

            if (!binary_pred(temp_key, key))
                // reset sum
                next = init;  
                
            *result = next;  
            next = binary_op(next, temp_value);

            temp_key = key;
        }
    }

    return result;
}

} // end namespace cpp
} // end namespace backend
} // end namespace detail
} // end namespace thrust

