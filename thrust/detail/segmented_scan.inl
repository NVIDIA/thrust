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


/*! \file segmented_scan.inl
 *  \brief Inline file for segmented_scan.h.
 */

#include <thrust/detail/dispatch/segmented_scan.h>

#include <thrust/iterator/iterator_traits.h>
#include <thrust/functional.h>

namespace thrust
{

namespace experimental
{

template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator>
  OutputIterator inclusive_segmented_scan(InputIterator1 first1,
                                          InputIterator1 last1,
                                          InputIterator2 first2,
                                          OutputIterator result)
{
    typedef typename thrust::iterator_traits<OutputIterator>::value_type OutputType;

    // assume plus as the associative operator
    return thrust::experimental::inclusive_segmented_scan(first1, last1, first2, result, thrust::plus<OutputType>());
}

template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename AssociativeOperator>
  OutputIterator inclusive_segmented_scan(InputIterator1 first1,
                                          InputIterator1 last1,
                                          InputIterator2 first2,
                                          OutputIterator result,
                                          AssociativeOperator binary_op)
{
    typedef typename thrust::iterator_traits<InputIterator2>::value_type KeyType;

    // assume equal_to as the binary predicate
    return thrust::experimental::inclusive_segmented_scan(first1, last1, first2, result, binary_op, thrust::equal_to<KeyType>());
}

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
    // dispatch on space
    return thrust::detail::dispatch::inclusive_segmented_scan(first1, last1, first2, result, binary_op, pred,
            typename thrust::iterator_space<InputIterator1>::type(),
            typename thrust::iterator_space<InputIterator2>::type(),
            typename thrust::iterator_space<OutputIterator>::type());
}

template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator>
  OutputIterator exclusive_segmented_scan(InputIterator1 first1,
                                          InputIterator1 last1,
                                          InputIterator2 first2,
                                          OutputIterator result)
{
    typedef typename thrust::iterator_traits<OutputIterator>::value_type OutputType;

    // assume 0 as the initialization value
    return thrust::experimental::exclusive_segmented_scan(first1, last1, first2, result, OutputType(0));
}

template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename T>
  OutputIterator exclusive_segmented_scan(InputIterator1 first1,
                                          InputIterator1 last1,
                                          InputIterator2 first2,
                                          OutputIterator result,
                                          const T init)
{
    typedef typename thrust::iterator_traits<OutputIterator>::value_type OutputType;

    // assume plus as the associative operator
    return thrust::experimental::exclusive_segmented_scan(first1, last1, first2, result, init, thrust::plus<OutputType>());
} 

template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename T,
         typename AssociativeOperator>
  OutputIterator exclusive_segmented_scan(InputIterator1 first1,
                                          InputIterator1 last1,
                                          InputIterator2 first2,
                                          OutputIterator result,
                                          const T init,
                                          AssociativeOperator binary_op)
{
    typedef typename thrust::iterator_traits<InputIterator2>::value_type KeyType;

    // assume equal_to as the binary predicate
    return thrust::experimental::exclusive_segmented_scan(first1, last1, first2, result, init, binary_op, thrust::equal_to<KeyType>());
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
    // dispatch on space
    return thrust::detail::dispatch::exclusive_segmented_scan(first1, last1, first2, result, init, binary_op, pred,
            typename thrust::iterator_space<InputIterator1>::type(),
            typename thrust::iterator_space<InputIterator2>::type(),
            typename thrust::iterator_space<OutputIterator>::type());
}

} // end namespace experimental

} // end namespace thrust

