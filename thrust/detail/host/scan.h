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
    typedef typename thrust::iterator_traits<OutputIterator>::value_type OutputType;

    if(first != last)
    {
        OutputType sum = *first;

        *result = sum;

        for(++first, ++result; first != last; ++first, ++result)
            *result = sum = binary_op(sum, *first);
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
    typedef typename thrust::iterator_traits<OutputIterator>::value_type OutputType;

    if(first != last)
    {
        OutputType tmp = *first;  // temporary value allows in-situ scan
        OutputType sum =  init;

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

} // end namespace host

} // end namespace detail

} // end namespace thrust

