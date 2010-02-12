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


/*! \file segmenteD_scan.h
 *  \brief Dispatch layer for the segmented scan functions.
 */

#pragma once

#include <thrust/iterator/iterator_traits.h>

#include <thrust/detail/host/segmented_scan.h>
#include <thrust/detail/device/segmented_scan.h>

namespace thrust
{

namespace detail
{

namespace dispatch
{

////////////////
// Host Paths //
////////////////

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
                                          BinaryPredicate pred,
                                          thrust::host_space_tag,
                                          thrust::host_space_tag,
                                          thrust::host_space_tag)
{
    return thrust::detail::host::inclusive_segmented_scan(first1, last1, first2, result, binary_op, pred);
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
                                          BinaryPredicate pred,
                                          thrust::host_space_tag,
                                          thrust::host_space_tag,
                                          thrust::host_space_tag)
{
    return thrust::detail::host::exclusive_segmented_scan(first1, last1, first2, result, init, binary_op, pred);
}


//////////////////
// Device Paths //
//////////////////

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
                                          BinaryPredicate pred,
                                          thrust::device_space_tag,
                                          thrust::device_space_tag,
                                          thrust::device_space_tag)
{
    return thrust::detail::device::inclusive_segmented_scan(first1, last1, first2, result, binary_op, pred);
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
                                          BinaryPredicate pred,
                                          thrust::device_space_tag,
                                          thrust::device_space_tag,
                                          thrust::device_space_tag)
{
    return thrust::detail::device::exclusive_segmented_scan(first1, last1, first2, result, init, binary_op, pred);
}

} // end namespace dispatch

} // end namespace detail

} // end namespace thrust

