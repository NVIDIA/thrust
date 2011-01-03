/*
 *  Copyright 2008-2011 NVIDIA Corporation
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


/*! \file equal.inl
 *  \brief Inline file for equal.h.
 */

#include <thrust/equal.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/mismatch.h>
#include <thrust/detail/internal_functional.h>

namespace thrust
{

template <typename InputIterator1, typename InputIterator2>
bool equal(InputIterator1 first1, InputIterator1 last1,
           InputIterator2 first2)
{
    typedef typename thrust::iterator_traits<InputIterator1>::value_type InputType1;

    return thrust::equal(first1, last1, first2, thrust::detail::equal_to<InputType1>());
}

template <typename InputIterator1, typename InputIterator2, 
          typename BinaryPredicate>
bool equal(InputIterator1 first1, InputIterator1 last1,
           InputIterator2 first2, BinaryPredicate binary_pred)
{
    return thrust::mismatch(first1, last1, first2, binary_pred).first == last1;
}

} // end namespace thrust

