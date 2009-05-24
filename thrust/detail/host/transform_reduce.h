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


/*! \file transform_reduce.h
 *  \brief Host implementation transform_reduce.
 */

#pragma once

namespace thrust
{

namespace detail
{

namespace host
{


template<typename InputIterator, 
         typename UnaryFunction, 
         typename OutputType,
         typename BinaryFunction>
  OutputType transform_reduce(InputIterator begin,
                              InputIterator end,
                              UnaryFunction unary_op,
                              OutputType init,
                              BinaryFunction binary_op)
{
    // initialize the result
    OutputType result = init;

    while(begin != end)
    {
        result = binary_op(result, unary_op(*begin));
        begin++;
    } // end while

    return result;
}


} // end namespace host

} // end namespace detail

} // end namespace thrust

