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


/*! \file scan.h
 *  \brief Host implementations of scan functions.
 */

#pragma once

#include <thrust/iterator/iterator_traits.h>

namespace thrust
{

namespace detail
{

namespace host
{

template<typename InputIterator,
         typename OutputIterator,
         typename AssociativeOperator>
  OutputIterator inclusive_scan(InputIterator first,
                                InputIterator last,
                                OutputIterator result,
                                AssociativeOperator binary_op)
{
    if(first != last) {
        typename thrust::iterator_traits<OutputIterator>::value_type prev;

        // first one is *first
        *result = prev = *first;

        for(++first, ++result;
                first != last;
                ++first, ++result)
        {
            *result = prev = binary_op(prev, *first);
        }
    }

    return result;
}


template<typename InputIterator,
         typename OutputIterator,
         typename T,
         typename AssociativeOperator>
  OutputIterator exclusive_scan(InputIterator first,
                                InputIterator last,
                                OutputIterator result,
                                T init,
                                AssociativeOperator binary_op)
{
    if(first != last) {
        typename thrust::iterator_traits<OutputIterator>::value_type temp = *first;
        typename thrust::iterator_traits<OutputIterator>::value_type next =  init;

        // first one is init
        *result = next;
        next = binary_op(init, temp);

        for(++first, ++result;
                first != last;
                ++first, ++result)
        {
            // use temp to permit in-place scans
            temp = *first;
            *result = next;
            next = binary_op(next, temp);
        }
    }

    return result;
} 

} // end namespace host

} // end namespace detail

} // end namespace thrust

