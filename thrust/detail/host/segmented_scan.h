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


/*! \file segmented_scan.h
 *  \brief Host implementations of segmented scan functions.
 */

#pragma once

#include <thrust/iterator/iterator_traits.h>

namespace thrust
{

namespace detail
{

namespace host
{

template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename AssociativeOperator,
         typename BinaryPredicate>
  OutputIterator inclusive_segmented_scan(InputIterator1 first1,
                                          InputIterator1 last1,
                                          InputIterator2 first2,
                                          OutputIterator result,
                                          AssociativeOperator binary_op,
                                          BinaryPredicate pred)
{
    if(first1 != last1) {
        typename thrust::iterator_traits<InputIterator2>::value_type prev_key;
        typename thrust::iterator_traits<OutputIterator>::value_type prev_value;

        // first one is *first
        prev_key = *first2;
        *result = prev_value = *first1;

        for(++first1, ++first2, ++result;
                first1 != last1;
                ++first1, ++first2, ++result)
        {
            typename thrust::iterator_traits<InputIterator2>::value_type key = *first2;

            if (pred(prev_key, key))
                *result = prev_value = binary_op(prev_value, *first1);
            else
                *result = prev_value = *first1;

            prev_key = key;
        }
    }

    return result;
}


template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename T,
         typename AssociativeOperator,
         typename BinaryPredicate>
  OutputIterator exclusive_segmented_scan(InputIterator1 first1,
                                          InputIterator1 last1,
                                          InputIterator2 first2,
                                          OutputIterator result,
                                          const T init,
                                          AssociativeOperator binary_op,
                                          BinaryPredicate pred)
{
    typedef typename thrust::iterator_traits<InputIterator2>::value_type KeyType;
    typedef typename thrust::iterator_traits<OutputIterator>::value_type ValueType;

    if(first1 != last1) {
        ValueType temp_value = *first1;
        KeyType   temp_key   = *first2;
        
        ValueType next = init;

        // first one is init
        *result = next;
        next = binary_op(next, temp_value);

        for(++first1, ++first2, ++result;
                first1 != last1;
                ++first1, ++first2, ++result)
        {
            // use temp to permit in-place scans
            temp_value = *first1;
            
            KeyType key = *first2;

            if (!pred(temp_key, key))
                // reset sum
                next = init;  
                
            *result = next;  
            next = binary_op(next, temp_value);

            temp_key = key;
        }
    }

    return result;
}


} // end namespace host

} // end namespace detail

} // end namespace thrust

