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
 *  \brief Host implementation remove functions.
 */

#pragma once

#include <algorithm>

namespace thrust
{

namespace detail
{

namespace host
{

template<typename ForwardIterator,
         typename InputIterator,
         typename Predicate>
  ForwardIterator remove_if(ForwardIterator first,
                            ForwardIterator last,
                            InputIterator stencil,
                            Predicate pred)
{
    // advance iterators until pred(stencil) is true we reach the end of input
    while(first != last && !bool(pred(*stencil)))
    {
        first++;
        stencil++;
    }

    if(first == last)
        return first;

    // result always trails first 
    ForwardIterator result = first;
    
    first++;
    stencil++;

    while(first != last)
    {
        if(!bool(pred(*stencil)))
        {
            *result = *first;
            result++;
        }
        first++;
        stencil++;
    }

    return result;
}

template<typename InputIterator,
         typename OutputIterator,
         typename Predicate>
  OutputIterator remove_copy_if(InputIterator first,
                                InputIterator last,
                                OutputIterator result,
                                Predicate pred)
{
    return std::remove_copy_if(first, last, result, pred);
}

template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename Predicate>
  OutputIterator remove_copy_if(InputIterator1 first,
                                InputIterator1 last,
                                InputIterator2 stencil,
                                OutputIterator result,
                                Predicate pred)
{
    while (first != last)
    {
        if (!bool(pred(*stencil)))
        {
            *result = *first;
            result++;
        }
        first++;
        stencil++;
    }

    return result;
}



} // last namespace host

} // last namespace detail

} // last namespace thrust

