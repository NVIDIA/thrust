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


/*! \file unique.h
 *  \brief C++ implementation of unique functions.
 */

#pragma once

#include <thrust/iterator/iterator_traits.h>
#include <thrust/pair.h>

namespace thrust
{
namespace detail
{
namespace backend
{
namespace cpp
{

template <typename InputIterator,
          typename OutputIterator,
          typename BinaryPredicate>
OutputIterator unique_copy(InputIterator first,
                           InputIterator last,
                           OutputIterator output,
                           BinaryPredicate binary_pred)
{
    typedef typename thrust::iterator_traits<InputIterator>::value_type T;

    if(first != last)
    {
        T prev = *first;
        
        for(++first; first != last; ++first)
        {
            T temp = *first;

            if (!binary_pred(prev, temp))
            {
                *output = prev;

                ++output;

                prev = temp;
            }
        }

        *output = prev;
        ++output;
    }

    return output;
}

template <typename ForwardIterator,
          typename BinaryPredicate>
ForwardIterator unique(ForwardIterator first,
                       ForwardIterator last,
                       BinaryPredicate binary_pred)
{
    // unique_copy() permits in-situ operation
    return thrust::detail::backend::cpp::unique_copy(first, last, first, binary_pred);
}

template <typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator1,
          typename OutputIterator2,
          typename BinaryPredicate>
  thrust::pair<OutputIterator1,OutputIterator2>
  unique_by_key_copy(InputIterator1 keys_first, 
                     InputIterator1 keys_last,
                     InputIterator2 values_first,
                     OutputIterator1 keys_output,
                     OutputIterator2 values_output,
                     BinaryPredicate binary_pred)
{
    typedef typename thrust::iterator_traits<InputIterator1>::value_type  InputKeyType;
    typedef typename thrust::iterator_traits<OutputIterator2>::value_type OutputValueType;

    if(keys_first != keys_last)
    {
        InputKeyType    temp_key   = *keys_first;
        OutputValueType temp_value = *values_first;
        
        for(++keys_first, ++values_first;
                keys_first != keys_last;
                ++keys_first, ++values_first)
        {
            InputKeyType    key   = *keys_first;
            OutputValueType value = *values_first;

            if (!binary_pred(temp_key, key))
            {
                *keys_output   = temp_key;
                *values_output = temp_value;

                ++keys_output;
                ++values_output;

                temp_key   = key;
                temp_value = value;
            }
        }

        *keys_output   = temp_key;
        *values_output = temp_value;

        ++keys_output;
        ++values_output;
    }
        
    return thrust::make_pair(keys_output, values_output);
}

template <typename ForwardIterator1,
          typename ForwardIterator2,
          typename BinaryPredicate>
  thrust::pair<ForwardIterator1,ForwardIterator2>
  unique_by_key(ForwardIterator1 keys_first, 
                ForwardIterator1 keys_last,
                ForwardIterator2 values_first,
                BinaryPredicate binary_pred)
{
    // unique_by_key_copy() permits in-situ operation
    return thrust::detail::backend::cpp::unique_by_key_copy(keys_first, keys_last, values_first, keys_first, values_first, binary_pred);
}

} // end namespace cpp
} // end namespace backend 
} // end namespace detail
} // end namespace thrust

