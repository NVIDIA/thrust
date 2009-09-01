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
              thrust::host_space_tag,
              thrust::host_space_tag,
              thrust::host_space_tag)
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
                 thrust::host_space_tag,
                 thrust::host_space_tag,
                 thrust::host_space_tag,
                 thrust::host_space_tag)
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
              thrust::device_space_tag,
              thrust::device_space_tag,
              thrust::device_space_tag)
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
                 thrust::device_space_tag,
                 thrust::device_space_tag,
                 thrust::device_space_tag,
                 thrust::device_space_tag)
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
              thrust::host_space_tag,    // destination
              thrust::device_space_tag,  // map
              thrust::device_space_tag)  // source
{
  // gather on device and transfer to host
  typedef typename thrust::iterator_traits<ForwardIterator>::value_type OutputType;
  raw_device_buffer<OutputType> buffer(thrust::distance(first,last));
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
              thrust::host_space_tag,   // destination
              thrust::host_space_tag,   // map
              thrust::device_space_tag) // source
{
  // move map to device and try again
  typedef typename thrust::iterator_traits<InputIterator>::value_type IndexType;
  raw_device_buffer<IndexType> d_map(thrust::distance(first,last));
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
              thrust::device_space_tag, // destination
              thrust::host_space_tag,   // map
              thrust::host_space_tag)   // destination
{
  // gather on host and transfer to device
  typedef typename thrust::iterator_traits<ForwardIterator>::value_type OutputType;
  raw_host_buffer<OutputType> buffer(thrust::distance(first,last));
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
              thrust::device_space_tag, // destination
              thrust::device_space_tag, // map
              thrust::host_space_tag)   // destination
{
  // move map to host and try again
  typedef typename thrust::iterator_traits<InputIterator>::value_type IndexType;
  raw_host_buffer<IndexType> h_map(thrust::distance(first,last));
  thrust::copy(map, map + (last - first), h_map.begin());
  thrust::gather(first, last, h_map.begin(), input);
} // end gather()


} // end dispatch

} // end detail

} // end thrust

