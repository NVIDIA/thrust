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


/*! \file scan.inl
 *  \brief Inline file for scan.h.
 */

#include <thrust/detail/dispatch/scan.h>

#include <thrust/iterator/iterator_traits.h>
#include <thrust/functional.h>

namespace thrust
{

//////////////////
// Entry Points //
//////////////////
template<typename InputIterator,
         typename OutputIterator>
  OutputIterator inclusive_scan(InputIterator first,
                                InputIterator last,
                                OutputIterator result)
{
    typedef typename thrust::iterator_traits<OutputIterator>::value_type OutputType;

    // assume plus as the associative operator
    return thrust::inclusive_scan(first, last, result, thrust::plus<OutputType>());
} 

template<typename InputIterator,
         typename OutputIterator,
         typename AssociativeOperator>
  OutputIterator inclusive_scan(InputIterator first,
                                InputIterator last,
                                OutputIterator result,
                                AssociativeOperator binary_op)
{
    // dispatch on space
    return thrust::detail::dispatch::inclusive_scan(first, last, result, binary_op,
            typename thrust::iterator_space<InputIterator>::type(),
            typename thrust::iterator_space<OutputIterator>::type());
}

template<typename InputIterator,
         typename OutputIterator>
  OutputIterator exclusive_scan(InputIterator first,
                                InputIterator last,
                                OutputIterator result)
{
    typedef typename thrust::iterator_traits<OutputIterator>::value_type OutputType;

    // assume 0 as the initialization value
    return thrust::exclusive_scan(first, last, result, OutputType(0));
}

template<typename InputIterator,
         typename OutputIterator,
         typename T>
  OutputIterator exclusive_scan(InputIterator first,
                                InputIterator last,
                                OutputIterator result,
                                T init)
{
    typedef typename thrust::iterator_traits<OutputIterator>::value_type OutputType;

    // assume plus as the associative operator
    return thrust::exclusive_scan(first, last, result, init, thrust::plus<OutputType>());
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
    // dispatch on space
    return thrust::detail::dispatch::exclusive_scan(first, last, result, init, binary_op,
            typename thrust::iterator_space<InputIterator>::type(),
            typename thrust::iterator_space<OutputIterator>::type());
}

/////////////////////
// Key-Value Scans //
/////////////////////

template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator>
  OutputIterator inclusive_scan_by_key(InputIterator1 first1,
                                       InputIterator1 last1,
                                       InputIterator2 first2,
                                       OutputIterator result)
{
    typedef typename thrust::iterator_traits<InputIterator1>::value_type InputType1;
    return thrust::inclusive_scan_by_key(first1, last1, first2, result, thrust::equal_to<InputType1>());
}

template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename BinaryPredicate>
  OutputIterator inclusive_scan_by_key(InputIterator1 first1,
                                       InputIterator1 last1,
                                       InputIterator2 first2,
                                       OutputIterator result,
                                       BinaryPredicate binary_pred)
{
    typedef typename thrust::iterator_traits<OutputIterator>::value_type OutputType;
    return thrust::inclusive_scan_by_key(first1, last1, first2, result, binary_pred, thrust::plus<OutputType>());
}

template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename BinaryPredicate,
         typename AssociativeOperator>
  OutputIterator inclusive_scan_by_key(InputIterator1 first1,
                                       InputIterator1 last1,
                                       InputIterator2 first2,
                                       OutputIterator result,
                                       BinaryPredicate binary_pred,
                                       AssociativeOperator binary_op)
{
    // dispatch on space
    return thrust::detail::dispatch::inclusive_scan_by_key
        (first1, last1, first2, result, binary_pred, binary_op,
            typename thrust::iterator_space<InputIterator1>::type(),
            typename thrust::iterator_space<InputIterator2>::type(),
            typename thrust::iterator_space<OutputIterator>::type());
}


template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator>
  OutputIterator exclusive_scan_by_key(InputIterator1 first1,
                                       InputIterator1 last1,
                                       InputIterator2 first2,
                                       OutputIterator result)
{
    typedef typename thrust::iterator_traits<OutputIterator>::value_type OutputType;
    return thrust::exclusive_scan_by_key(first1, last1, first2, result, OutputType(0));
}

template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename T>
  OutputIterator exclusive_scan_by_key(InputIterator1 first1,
                                       InputIterator1 last1,
                                       InputIterator2 first2,
                                       OutputIterator result,
                                       T init)
{
    typedef typename thrust::iterator_traits<InputIterator1>::value_type InputType1;
    return thrust::exclusive_scan_by_key(first1, last1, first2, result, init, thrust::equal_to<InputType1>());
}

template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename T,
         typename BinaryPredicate>
  OutputIterator exclusive_scan_by_key(InputIterator1 first1,
                                       InputIterator1 last1,
                                       InputIterator2 first2,
                                       OutputIterator result,
                                       T init,
                                       BinaryPredicate binary_pred)
{
    typedef typename thrust::iterator_traits<OutputIterator>::value_type OutputType;
    return thrust::exclusive_scan_by_key(first1, last1, first2, result, init, binary_pred, thrust::plus<OutputType>());
}

template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename T,
         typename BinaryPredicate,
         typename AssociativeOperator>
  OutputIterator exclusive_scan_by_key(InputIterator1 first1,
                                       InputIterator1 last1,
                                       InputIterator2 first2,
                                       OutputIterator result,
                                       T init,
                                       BinaryPredicate binary_pred,
                                       AssociativeOperator binary_op)
{
    // dispatch on space
    return thrust::detail::dispatch::exclusive_scan_by_key
        (first1, last1, first2, result, init, binary_pred, binary_op,
            typename thrust::iterator_space<InputIterator1>::type(),
            typename thrust::iterator_space<InputIterator2>::type(),
            typename thrust::iterator_space<OutputIterator>::type());
}

} // end namespace thrust

