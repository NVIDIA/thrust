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


/*! \file transform_scan.h
 *  \brief Defines the host implementations of 
 *         the family oftransform scan functions.
 */

#pragma once

namespace thrust
{

namespace detail
{

namespace host
{

template<typename InputIterator,
         typename OutputIterator,
         typename UnaryFunction,
         typename AssociativeOperator>
  void transform_inclusive_scan(InputIterator begin,
                                InputIterator end,
                                OutputIterator result,
                                UnaryFunction unary_op,
                                AssociativeOperator binary_op)
{
    if(begin != end) {
        typename thrust::iterator_traits<OutputIterator>::value_type last;

        // first one is *begin
        *result = last = unary_op(*begin);

        for(++begin, ++result;
                begin != end;
                ++begin, ++result)
        {
            *result = last = binary_op(last, unary_op(*begin));
        }
    }
}


template<typename InputIterator,
         typename OutputIterator,
         typename UnaryFunction,
         typename T,
         typename AssociativeOperator>
  void transform_exclusive_scan(InputIterator begin,
                                InputIterator end,
                                OutputIterator result,
                                UnaryFunction unary_op,
                                T init,
                                AssociativeOperator binary_op)
{
    if(begin != end) {
        typename thrust::iterator_traits<OutputIterator>::value_type temp = *begin;
        typename thrust::iterator_traits<OutputIterator>::value_type next =  init;

        // first one is init
        *result = next;
        next = binary_op(init, unary_op(temp));

        for(++begin, ++result;
                begin != end;
                ++begin, ++result)
        {
            // use temp to permit in-place scans
            temp = unary_op(*begin);
            *result = next;
            next = binary_op(next, temp);
        }
    }
} 


} // end namespace host

} // end namespace detail

} // end namespace thrust

