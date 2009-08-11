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
 *  \brief Defines the interface to the
 *         dispatch layer of the gather function.
 */

#pragma once

#include <thrust/copy.h>
#include <thrust/iterator/iterator_traits.h>

#include <thrust/detail/host/gather.h>
#include <thrust/detail/device/gather.h>

#include <thrust/distance.h>
#include <thrust/detail/raw_buffer.h>

namespace thrust
{

namespace detail
{

namespace dispatch
{

/////////////////////////////
// From Host To Host Paths //
/////////////////////////////
template<typename ForwardIterator,
         typename InputIterator,
         typename RandomAccessIterator>
  void gather(ForwardIterator first,
              ForwardIterator last,
              InputIterator map,
              RandomAccessIterator input,
              thrust::experimental::space::host,
              thrust::experimental::space::host,
              thrust::experimental::space::host)
{
    thrust::detail::host::gather(first, last, map, input);
}

template<typename ForwardIterator,
         typename InputIterator1,
         typename InputIterator2,
         typename RandomAccessIterator,
         typename Predicate>
  void gather_if(ForwardIterator first,
                 ForwardIterator last,
                 InputIterator1 map,
                 InputIterator2 stencil,
                 RandomAccessIterator input,
                 Predicate pred,
                 thrust::experimental::space::host,
                 thrust::experimental::space::host,
                 thrust::experimental::space::host,
                 thrust::experimental::space::host)
{
    thrust::detail::host::gather_if(first, last, map, stencil, input, pred);
}


/////////////////////////////////
// From Device To Device Paths //
/////////////////////////////////
template<typename ForwardIterator,
         typename InputIterator,
         typename RandomAccessIterator>
  void gather(ForwardIterator first,
              ForwardIterator last,
              InputIterator  map,
              RandomAccessIterator input,
              thrust::experimental::space::device,
              thrust::experimental::space::device,
              thrust::experimental::space::device)
{
    thrust::detail::device::gather(first, last, map, input);
}


template<typename ForwardIterator,
         typename InputIterator1,
         typename InputIterator2,
         typename RandomAccessIterator,
         typename Predicate>
  void gather_if(ForwardIterator first,
                 ForwardIterator last,
                 InputIterator1 map,
                 InputIterator2 stencil,
                 RandomAccessIterator input,
                 Predicate pred,
                 thrust::experimental::space::device,
                 thrust::experimental::space::device,
                 thrust::experimental::space::device,
                 thrust::experimental::space::device)
{
    thrust::detail::device::gather_if(first, last, map, stencil, input, pred);
}


///////////////////////////////
// From Device To Host Paths //
///////////////////////////////

template<typename ForwardIterator,
         typename InputIterator,
         typename RandomAccessIterator>
  void gather(ForwardIterator first,
              ForwardIterator last,
              InputIterator map,
              RandomAccessIterator input,
              thrust::experimental::space::host,    // destination
              thrust::experimental::space::device,  // map
              thrust::experimental::space::device)  // source
{
  // gather on device and transfer to host
  typedef typename thrust::iterator_traits<ForwardIterator>::value_type OutputType;
  raw_buffer<OutputType, experimental::space::device> buffer(thrust::distance(first,last));
  thrust::gather(buffer.begin(), buffer.end(), map, input);
  thrust::copy(buffer.begin(), buffer.end(), first);
} // end gather()

template<typename ForwardIterator,
         typename InputIterator,
         typename RandomAccessIterator>
  void gather(ForwardIterator first,
              ForwardIterator last,
              InputIterator map,
              RandomAccessIterator input,
              thrust::experimental::space::host,   // destination
              thrust::experimental::space::host,   // map
              thrust::experimental::space::device) // source
{
  // move map to device and try again
  typedef typename thrust::iterator_traits<InputIterator>::value_type IndexType;
  raw_buffer<IndexType, experimental::space::device> d_map(thrust::distance(first,last));
  thrust::copy(map, map + (last - first), d_map.begin());
  thrust::gather(first, last, d_map.begin(), input);
} // end gather()


///////////////////////////////
// From Host To Device Paths //
///////////////////////////////

template<typename ForwardIterator,
         typename InputIterator,
         typename RandomAccessIterator>
  void gather(ForwardIterator first,
              ForwardIterator last,
              InputIterator map,
              RandomAccessIterator input,
              thrust::experimental::space::device, // destination
              thrust::experimental::space::host,   // map
              thrust::experimental::space::host)   // destination
{
  // gather on host and transfer to device
  typedef typename thrust::iterator_traits<ForwardIterator>::value_type OutputType;
  raw_buffer<OutputType,experimental::space::host> buffer(thrust::distance(first,last));
  thrust::gather(buffer.begin(), buffer.end(), map, input);
  thrust::copy(buffer.begin(), buffer.end(), first);
} // end gather()

template<typename ForwardIterator,
         typename InputIterator,
         typename RandomAccessIterator>
  void gather(ForwardIterator first,
              ForwardIterator last,
              InputIterator map,
              RandomAccessIterator input,
              thrust::experimental::space::device, // destination
              thrust::experimental::space::device, // map
              thrust::experimental::space::host)   // destination
{
  // move map to host and try again
  typedef typename thrust::iterator_traits<InputIterator>::value_type IndexType;
  raw_buffer<IndexType, experimental::space::host> h_map(thrust::distance(first,last));
  thrust::copy(map, map + (last - first), h_map.begin());
  thrust::gather(first, last, h_map.begin(), input);
} // end gather()


} // end dispatch

} // end detail

} // end thrust

