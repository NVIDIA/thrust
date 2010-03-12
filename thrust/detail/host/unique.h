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


/*! \file remove.h
 *  \brief Host implementation unique functions.
 */

#pragma once

#include <algorithm>

#include <thrust/iterator/iterator_traits.h>

namespace thrust
{
namespace detail
{
namespace host
{

template <typename ForwardIterator,
          typename BinaryPredicate>
ForwardIterator unique(ForwardIterator first,
                       ForwardIterator last,
                       BinaryPredicate binary_pred)
{
    return std::unique(first, last, binary_pred);
}

template <typename InputIterator,
          typename OutputIterator,
          typename BinaryPredicate>
OutputIterator unique_copy(InputIterator first,
                           InputIterator last,
                           OutputIterator output,
                           BinaryPredicate binary_pred)
{
    return std::unique_copy(first, last, output, binary_pred);
}

template <typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator1,
          typename OutputIterator2,
          typename BinaryPredicate>
  thrust::pair<OutputIterator1,OutputIterator2>
  unique_copy_by_key(InputIterator1 keys_first, 
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
    // unique_copy_by_key() permits in-situ operation
    return thrust::detail::host::unique_copy_by_key
        (keys_first, keys_last, values_first, keys_first, values_first, binary_pred);
}

} // last namespace host
} // last namespace detail
} // last namespace thrust

