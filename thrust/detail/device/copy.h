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


/*! \file copy.h
 *  \brief Device implementations for copy.
 */

#pragma once

#include <thrust/device_ptr.h>

namespace thrust
{

// forward declare these here before the #includes to WAR 
// some issues with recursive #includes

inline void device_free(thrust::device_ptr<void> ptr);

inline thrust::device_ptr<void> device_malloc(const std::size_t n);

template<typename T>
  inline thrust::device_ptr<T> device_malloc(const std::size_t n);

} // end thrust

#include <thrust/iterator/iterator_traits.h>
#include <thrust/functional.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>
#include <thrust/distance.h>
#include <thrust/transform_scan.h>
#include <thrust/scatter.h>

namespace thrust
{

namespace detail
{

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
  typedef typename thrust::iterator_traits<RandomAccessDeviceIterator1>::difference_type difference_type;

  difference_type n = thrust::distance(first, last);

  difference_type size_of_new_sequence = 0;
  if(n > 0)
  {
    // scan pred(stencil) to a temp buffer
    thrust::device_ptr<difference_type> pred_scatter_indices = thrust::device_malloc<difference_type>(n);
    thrust::transform_exclusive_scan(stencil,
                                     stencil + n,
                                     pred_scatter_indices,
                                     pred,
                                     static_cast<unsigned int>(0),
                                     thrust::plus<unsigned int>());

    // scatter the true elements
    thrust::scatter_if(first,
                       last,
                       pred_scatter_indices,
                       stencil,
                       result,
                       pred);

    // find the end of the new sequence
    size_of_new_sequence = pred_scatter_indices[n - 1]
                         + (pred(*(stencil + (n-1))) ? 1 : 0);

    thrust::device_free(pred_scatter_indices);
  } // end if

  return result + size_of_new_sequence;
} // end copy_if()

} // end device

} // end detail

} // end thrust

