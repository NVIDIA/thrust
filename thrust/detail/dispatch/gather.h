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
#include <thrust/functional.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/iterator_categories.h>

#include <thrust/detail/host/gather.h>
#include <thrust/detail/device/cuda/gather.h>

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
template<typename ForwardIterator,
         typename InputIterator,
         typename RandomAccessIterator>
  void gather(ForwardIterator first,
              ForwardIterator last,
              InputIterator map,
              RandomAccessIterator input,
              thrust::forward_host_iterator_tag,
              thrust::input_host_iterator_tag,
              thrust::random_access_host_iterator_tag)
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
                 thrust::forward_host_iterator_tag,
                 thrust::input_host_iterator_tag,
                 thrust::input_host_iterator_tag,
                 thrust::random_access_host_iterator_tag)
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
              thrust::random_access_device_iterator_tag,
              thrust::random_access_device_iterator_tag,
              thrust::random_access_device_iterator_tag)
{
    thrust::detail::device::cuda::gather(first, last, map, input);
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
                 thrust::random_access_device_iterator_tag,
                 thrust::random_access_device_iterator_tag,
                 thrust::random_access_device_iterator_tag,
                 thrust::random_access_device_iterator_tag)
{
    thrust::detail::device::cuda::gather_if(first, last, map, stencil, input, pred);
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
              thrust::forward_host_iterator_tag,          // destination
              thrust::random_access_device_iterator_tag,  // map
              thrust::random_access_device_iterator_tag)  // source
{
  // gather on device and transfer to host
  typedef typename thrust::iterator_traits<ForwardIterator>::value_type OutputType;
  thrust::device_ptr<OutputType> buffer = thrust::device_malloc<OutputType>(last - first);
  thrust::gather(buffer, buffer + (last - first), map, input);
  thrust::copy(buffer, buffer + (last - first), first);
  thrust::device_free(buffer);
} // end gather()

template<typename ForwardIterator,
         typename InputIterator,
         typename RandomAccessIterator>
  void gather(ForwardIterator first,
              ForwardIterator last,
              InputIterator map,
              RandomAccessIterator input,
              thrust::forward_host_iterator_tag,         // destination
              thrust::input_host_iterator_tag,           // map
              thrust::random_access_device_iterator_tag) // source
{
  // move map to device and try again
  typedef typename thrust::iterator_traits<InputIterator>::value_type IndexType;
  thrust::device_ptr<IndexType> d_map = thrust::device_malloc<IndexType>(last - first);
  thrust::copy(map, map + (last - first), d_map);
  thrust::gather(first, last, d_map, input);
  thrust::device_free(d_map);
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
              thrust::random_access_device_iterator_tag, // destination
              thrust::input_host_iterator_tag,           // map
              thrust::random_access_host_iterator_tag)   // destination
{
  // gather on host and transfer to device
  typedef typename thrust::iterator_traits<ForwardIterator>::value_type OutputType;
  OutputType * buffer = (OutputType *) malloc( (last - first) * sizeof(OutputType) ); // XXX replace with host_malloc
  thrust::gather(buffer, buffer + (last - first), map, input);
  thrust::copy(buffer, buffer + (last - first), first);
  free(buffer);
} // end gather()

template<typename ForwardIterator,
         typename InputIterator,
         typename RandomAccessIterator>
  void gather(ForwardIterator first,
              ForwardIterator last,
              InputIterator map,
              RandomAccessIterator input,
              thrust::random_access_device_iterator_tag, // destination
              thrust::random_access_device_iterator_tag, // map
              thrust::random_access_host_iterator_tag)   // destination
{
  // move map to host and try again
  typedef typename thrust::iterator_traits<InputIterator>::value_type IndexType;
  IndexType * h_map = (IndexType *) malloc( (last - first) * sizeof(IndexType) ); // XXX replace with host_malloc
  thrust::copy(map, map + (last - first), h_map);
  thrust::gather(first, last, h_map, input);
  free(h_map);
} // end gather()


} // end dispatch

} // end detail

} // end thrust

