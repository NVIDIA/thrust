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


/*! \file mismatch.inl
 *  \brief Inline file for mismatch.h
 */

#include <thrust/functional.h>
#include <thrust/iterator/iterator_traits.h>

#include <thrust/detail/dispatch/mismatch.h>

namespace thrust
{

template <typename InputIterator1, typename InputIterator2>
thrust::pair<InputIterator1, InputIterator2> mismatch(InputIterator1 first1,
                                                      InputIterator1 last1,
                                                      InputIterator2 first2)
{
    typedef typename thrust::iterator_traits<InputIterator1>::value_type InputType1;

    return thrust::mismatch(first1, last1, first2, thrust::detail::equal_to<InputType1>());
}

template <typename InputIterator1, typename InputIterator2, typename BinaryPredicate>
thrust::pair<InputIterator1, InputIterator2> mismatch(InputIterator1 first1,
                                                      InputIterator1 last1,
                                                      InputIterator2 first2,
                                                      BinaryPredicate pred)
{
    return thrust::detail::dispatch::mismatch(first1, last1, first2, pred,
            typename thrust::iterator_space<InputIterator1>::type(),
            typename thrust::iterator_space<InputIterator2>::type());
}

} // end namespace thrust

