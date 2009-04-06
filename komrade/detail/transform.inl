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


/*! \file transform.inl
 *  \brief Inline file for transform.h.
 */

#include <komrade/transform.h>
#include <komrade/iterator/iterator_traits.h>
#include <komrade/detail/dispatch/transform.h>

namespace komrade
{

template<typename InputIterator,
         typename OutputIterator,
         typename UnaryFunction>
  OutputIterator transform(InputIterator first, InputIterator last,
                           OutputIterator result,
                           UnaryFunction op)
{
    // dispatch on category
    return komrade::detail::dispatch::transform(first, last, result, op,
            typename komrade::iterator_traits<InputIterator>::iterator_category(),
            typename komrade::iterator_traits<OutputIterator>::iterator_category());
} // end transform()


template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename BinaryFunction>
  OutputIterator transform(InputIterator1 first1, InputIterator1 last1,
                           InputIterator2 first2,
                           OutputIterator result,
                           BinaryFunction op)
{
    // dispatch on category
    return komrade::detail::dispatch::transform(first1, last1, first2, result, op,
            typename komrade::iterator_traits<InputIterator1>::iterator_category(),
            typename komrade::iterator_traits<InputIterator2>::iterator_category(),
            typename komrade::iterator_traits<OutputIterator>::iterator_category());
} // end transform()

} // end namespace komrade

