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


/*! \file transform.h
 *  \brief Host implementation of predicated_transform.
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
         typename UnaryFunction>
  OutputIterator transform(InputIterator first, InputIterator last,
                           OutputIterator result,
                           UnaryFunction unary_op)
{
    for(; first != last; ++first, ++result)
    {
        *result = unary_op(*first);
    }

    return result;
} // end transform()

template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename BinaryFunction>
  OutputIterator transform(InputIterator1 first1, InputIterator1 last1,
                           InputIterator2 first2,
                           OutputIterator result,
                           BinaryFunction binary_op)
{
  for(; first1 != last1; ++first1, ++first2, ++result)
  {
      *result = binary_op(*first1, *first2);
  }

  return result;
} // end transform()

template<typename InputIterator1,
         typename InputIterator2,
         typename ForwardIterator,
         typename UnaryFunction,
         typename Predicate>
  ForwardIterator transform_if(InputIterator1 first, InputIterator1 last,
                               InputIterator2 stencil,
                               ForwardIterator result,
                               UnaryFunction unary_op,
                               Predicate pred)
{
    for(; first != last; ++first, ++stencil, ++result)
    {
        if(pred(*stencil))
        {
            *result = unary_op(*first);
        }
    }

    return result;
} // end transform_if()


template<typename InputIterator1,
         typename InputIterator2,
         typename InputIterator3,
         typename ForwardIterator,
         typename BinaryFunction,
         typename Predicate>
  ForwardIterator transform_if(InputIterator1 first1, InputIterator1 last1,
                               InputIterator2 first2,
                               InputIterator3 stencil,
                               ForwardIterator result,
                               BinaryFunction binary_op,
                               Predicate pred)
{
    for(; first1 != last1; ++first1, ++first2, ++stencil, ++result)
    {
        if(pred(*stencil))
        {
            *result = binary_op(*first1, *first2);
        }
    }

    return result;
} // end transform_if()

} // end host

} // end detail

} // end thrust

