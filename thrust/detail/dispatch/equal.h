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


/*! \file gather.h
 *  \brief Defines the dispatch layer of the equal function.
 */

#pragma once

#include <thrust/iterator/iterator_traits.h>

#include <thrust/detail/host/equal.h>
#include <thrust/detail/device/equal.h>

#include <thrust/distance.h>
#include <thrust/detail/raw_buffer.h>

namespace thrust
{

namespace detail
{

namespace dispatch
{

//////////////////////
// Host<->Host Path //
//////////////////////
template <typename InputIterator1, typename InputIterator2, 
          typename BinaryPredicate>
bool equal(InputIterator1 first1, InputIterator1 last1,
           InputIterator2 first2, BinaryPredicate binary_pred,
           thrust::host_space_tag,
           thrust::host_space_tag)
{
    return thrust::detail::host::equal(first1, last1, first2, binary_pred);
}

//////////////////////////
// Device<->Device Path //
//////////////////////////
template <typename InputIterator1, typename InputIterator2, 
          typename BinaryPredicate>
bool equal(InputIterator1 first1, InputIterator1 last1,
           InputIterator2 first2, BinaryPredicate binary_pred,
           thrust::device_space_tag,
           thrust::device_space_tag)
{
    return thrust::detail::device::equal(first1, last1, first2, binary_pred);
}

////////////////////////
// Host<->Device Path //
////////////////////////
template <typename InputIterator1, typename InputIterator2, 
          typename BinaryPredicate>
bool equal(InputIterator1 first1, InputIterator1 last1,
           InputIterator2 first2, BinaryPredicate binary_pred,
           thrust::host_space_tag,
           thrust::device_space_tag)
{
    typedef typename thrust::iterator_traits<InputIterator2>::value_type InputType2;
    
    // copy device sequence to host and compare on host
    raw_host_buffer<InputType2> buffer(first2, first2 + thrust::distance(first1, last1));

    return thrust::detail::host::equal(first1, last1, buffer.begin(), binary_pred);
}
  
////////////////////////
// Device<->Host Path //
////////////////////////
template <typename InputIterator1, typename InputIterator2, 
          typename BinaryPredicate>
bool equal(InputIterator1 first1, InputIterator1 last1,
           InputIterator2 first2, BinaryPredicate binary_pred,
           thrust::device_space_tag,
           thrust::host_space_tag)
{
    typedef typename thrust::iterator_traits<InputIterator1>::value_type InputType1;
    
    // copy device sequence to host and compare on host
    raw_host_buffer<InputType1> buffer(first1, last1);

    return thrust::detail::host::equal(buffer.begin(), buffer.end(), first2, binary_pred);
}

} // end dispatch

} // end detail

} // end thrust

