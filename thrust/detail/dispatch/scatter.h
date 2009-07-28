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


/*! \file scatter.h
 *  \brief Dispatch layer for scatter functions
 */

#pragma once

#include <thrust/iterator/iterator_traits.h>

#include <thrust/detail/host/scatter.h>
#include <thrust/detail/device/scatter.h>

#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

namespace thrust
{

namespace detail
{

namespace dispatch
{


/////////////////////////////
// From Host To Host Paths //
/////////////////////////////
template<typename InputIterator1,
         typename InputIterator2,
         typename RandomAccessIterator>
  void scatter(InputIterator1 first,
               InputIterator1 last,
               InputIterator2 map,
               RandomAccessIterator output,
               thrust::experimental::space::host,
               thrust::experimental::space::host,
               thrust::experimental::space::host)
{
    thrust::detail::host::scatter(first, last, map, output);
}


template<typename InputIterator1,
         typename InputIterator2,
         typename InputIterator3,
         typename RandomAccessIterator,
         typename Predicate>
  void scatter_if(InputIterator1 first,
                  InputIterator1 last,
                  InputIterator2 map,
                  InputIterator3 stencil,
                  RandomAccessIterator output,
                  Predicate pred,
                  thrust::experimental::space::host,
                  thrust::experimental::space::host,
                  thrust::experimental::space::host,
                  thrust::experimental::space::host)
{
    thrust::detail::host::scatter_if(first, last, map, stencil, output, pred);
} 


/////////////////////////////////
// From Device To Device Paths //
/////////////////////////////////
template<typename InputIterator1,
         typename InputIterator2,
         typename RandomAccessIterator>
  void scatter(InputIterator1 first,
               InputIterator1 last,
               InputIterator2 map,
               RandomAccessIterator output,
               thrust::experimental::space::device,
               thrust::experimental::space::device,
               thrust::experimental::space::device)
{
    thrust::detail::device::scatter(first, last, map, output);
}


template<typename InputIterator1,
         typename InputIterator2,
         typename InputIterator3,
         typename RandomAccessIterator,
         typename Predicate>
  void scatter_if(InputIterator1 first,
                  InputIterator1 last,
                  InputIterator2 map,
                  InputIterator3 stencil,
                  RandomAccessIterator output,
                  Predicate pred,
                  thrust::experimental::space::device,
                  thrust::experimental::space::device,
                  thrust::experimental::space::device,
                  thrust::experimental::space::device)
{
    thrust::detail::device::scatter_if(first, last, map, stencil, output, pred);
}


///////////////////////////////
// From Device To Host Paths //
///////////////////////////////
template<typename InputIterator1,
         typename InputIterator2,
         typename RandomAccessIterator>
  void scatter(InputIterator1 first,
               InputIterator1 last,
               InputIterator2 map,
               RandomAccessIterator output,
               thrust::experimental::space::device,  // input on device
               thrust::experimental::space::host,    // map on host
               thrust::experimental::space::host)    // destination on host
{
  // copy input to host and scatter on host
  typedef typename thrust::iterator_traits<InputIterator1>::value_type InputType;
  InputType * buffer = (InputType *) malloc( (last - first) * sizeof(InputType) ); // XXX replace with host_malloc
  thrust::copy(first, last, buffer);
  thrust::scatter(buffer, buffer, map, output);
  free(buffer);
} // end scatter()


template<typename InputIterator1,
         typename InputIterator2,
         typename RandomAccessIterator>
  void scatter(InputIterator1 first,
               InputIterator1 last,
               InputIterator2 map,
               RandomAccessIterator output,
               thrust::experimental::space::device,  // input on device
               thrust::experimental::space::device,  // map on device
               thrust::experimental::space::host)    // destination on host
{
  // copy map to host and try again
  typedef typename thrust::iterator_traits<InputIterator2>::value_type IndexType;
  IndexType * h_map = (IndexType *) malloc( (last - first) * sizeof(IndexType) ); // XXX replace with host_malloc
  thrust::copy(map, map + (last - first), h_map);
  thrust::scatter(first, last, h_map, output);
  free(h_map);
} // end scatter()


///////////////////////////////
// From Host To Device Paths //
///////////////////////////////
template<typename InputIterator1,
         typename InputIterator2,
         typename RandomAccessIterator>
  void scatter(InputIterator1 first,
               InputIterator1 last,
               InputIterator2 map,
               RandomAccessIterator output,
               thrust::experimental::space::host,    // input on host
               thrust::experimental::space::device,  // map on device
               thrust::experimental::space::device)  // destination on device
{
  // copy input to device and scatter on device
  typedef typename thrust::iterator_traits<InputIterator1>::value_type InputType;
  thrust::device_ptr<InputType> buffer = thrust::device_malloc<InputType>(last - first);
  thrust::copy(first, last, buffer);
  thrust::scatter(buffer, buffer + (last - first), map, output);
  thrust::device_free(buffer);
} // end scatter()


template<typename InputIterator1,
         typename InputIterator2,
         typename RandomAccessIterator>
  void scatter(InputIterator1 first,
               InputIterator1 last,
               InputIterator2 map,
               RandomAccessIterator output,
               thrust::experimental::space::host,    // input on host
               thrust::experimental::space::host,    // map on host
               thrust::experimental::space::device)  // destination on device
{
  // copy map to device and try again
  typedef typename thrust::iterator_traits<InputIterator2>::value_type IndexType;
  thrust::device_ptr<IndexType> d_map = thrust::device_malloc<IndexType>(last - first);
  thrust::copy(map, map + (last - first), d_map);
  thrust::scatter(first, last, d_map, output);
  thrust::device_free(d_map);
} // end scatter()

} // end dispatch

} // end detail

} // end thrust

