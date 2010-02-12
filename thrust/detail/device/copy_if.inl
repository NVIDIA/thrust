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


/*! \file copy.h
 *  \brief Device implementations for copy.
 */

#pragma once

#include <thrust/iterator/iterator_traits.h>
#include <thrust/functional.h>
#include <thrust/distance.h>
#include <thrust/transform_scan.h>
#include <thrust/scatter.h>
#include <thrust/detail/raw_buffer.h>

namespace thrust
{

namespace detail
{

// forward declaration to WAR circular #inclusion
template<typename,typename> class raw_buffer;

namespace device
{

template<typename RandomAccessDeviceIterator1,
         typename RandomAccessDeviceIterator2,
         typename RandomAccessDeviceIterator3,
         typename Predicate>
   RandomAccessDeviceIterator3 copy_if(RandomAccessDeviceIterator1 first,
                                       RandomAccessDeviceIterator1 last,
                                       RandomAccessDeviceIterator2 stencil,
                                       RandomAccessDeviceIterator3 result,
                                       Predicate pred)
{
    if (first == last)
        return result;

    typedef typename thrust::iterator_traits<RandomAccessDeviceIterator1>::difference_type difference_type;

    difference_type n = thrust::distance(first, last);

    difference_type size_of_new_sequence = 0;

    // scan pred(stencil) to a temp buffer
    thrust::detail::raw_buffer<difference_type, device_space_tag> pred_scatter_indices(n);
    thrust::transform_exclusive_scan(stencil,
                                     stencil + n,
                                     pred_scatter_indices.begin(),
                                     pred,
                                     static_cast<unsigned int>(0),
                                     thrust::plus<unsigned int>());

    // scatter the true elements
    thrust::scatter_if(first,
                       last,
                       pred_scatter_indices.begin(),
                       stencil,
                       result,
                       pred);

    // find the end of the new sequence
    size_of_new_sequence = pred_scatter_indices[n - 1] + (pred(*(stencil + (n-1))) ? 1 : 0);

    return result + size_of_new_sequence;
} // end copy_if()

} // end device

} // end detail

} // end thrust

