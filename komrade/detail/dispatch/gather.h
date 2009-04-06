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

#include <komrade/copy.h>
#include <komrade/functional.h>
#include <komrade/iterator/iterator_traits.h>
#include <komrade/iterator/iterator_categories.h>

#include <komrade/detail/host/gather.h>
#include <komrade/detail/device/cuda/gather.h>

#include <komrade/device_malloc.h>
#include <komrade/device_free.h>

namespace komrade
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
              komrade::forward_host_iterator_tag,
              komrade::input_host_iterator_tag,
              komrade::random_access_host_iterator_tag)
{
    komrade::detail::host::gather(first, last, map, input);
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
                 komrade::forward_host_iterator_tag,
                 komrade::input_host_iterator_tag,
                 komrade::input_host_iterator_tag,
                 komrade::random_access_host_iterator_tag)
{
    komrade::detail::host::gather_if(first, last, map, stencil, input, pred);
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
              komrade::random_access_device_iterator_tag,
              komrade::random_access_device_iterator_tag,
              komrade::random_access_device_iterator_tag)
{
    komrade::detail::device::cuda::gather(first, last, map, input);
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
                 komrade::random_access_device_iterator_tag,
                 komrade::random_access_device_iterator_tag,
                 komrade::random_access_device_iterator_tag,
                 komrade::random_access_device_iterator_tag)
{
    komrade::detail::device::cuda::gather_if(first, last, map, stencil, input, pred);
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
              komrade::forward_host_iterator_tag,          // destination
              komrade::random_access_device_iterator_tag,  // map
              komrade::random_access_device_iterator_tag)  // source
{
  // gather on device and transfer to host
  typedef typename komrade::iterator_traits<ForwardIterator>::value_type OutputType;
  komrade::device_ptr<OutputType> buffer = komrade::device_malloc<OutputType>(last - first);
  komrade::gather(buffer, buffer + (last - first), map, input);
  komrade::copy(buffer, buffer + (last - first), first);
  komrade::device_free(buffer);
} // end gather()

template<typename ForwardIterator,
         typename InputIterator,
         typename RandomAccessIterator>
  void gather(ForwardIterator first,
              ForwardIterator last,
              InputIterator map,
              RandomAccessIterator input,
              komrade::forward_host_iterator_tag,         // destination
              komrade::input_host_iterator_tag,           // map
              komrade::random_access_device_iterator_tag) // source
{
  // move map to device and try again
  typedef typename komrade::iterator_traits<InputIterator>::value_type IndexType;
  komrade::device_ptr<IndexType> d_map = komrade::device_malloc<IndexType>(last - first);
  komrade::copy(map, map + (last - first), d_map);
  komrade::gather(first, last, d_map, input);
  komrade::device_free(d_map);
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
              komrade::random_access_device_iterator_tag, // destination
              komrade::input_host_iterator_tag,           // map
              komrade::random_access_host_iterator_tag)   // destination
{
  // gather on host and transfer to device
  typedef typename komrade::iterator_traits<ForwardIterator>::value_type OutputType;
  OutputType * buffer = (OutputType *) malloc( (last - first) * sizeof(OutputType) ); // XXX replace with host_malloc
  komrade::gather(buffer, buffer + (last - first), map, input);
  komrade::copy(buffer, buffer + (last - first), first);
  free(buffer);
} // end gather()

template<typename ForwardIterator,
         typename InputIterator,
         typename RandomAccessIterator>
  void gather(ForwardIterator first,
              ForwardIterator last,
              InputIterator map,
              RandomAccessIterator input,
              komrade::random_access_device_iterator_tag, // destination
              komrade::random_access_device_iterator_tag, // map
              komrade::random_access_host_iterator_tag)   // destination
{
  // move map to host and try again
  typedef typename komrade::iterator_traits<InputIterator>::value_type IndexType;
  IndexType * h_map = (IndexType *) malloc( (last - first) * sizeof(IndexType) ); // XXX replace with host_malloc
  komrade::copy(map, map + (last - first), h_map);
  komrade::gather(first, last, h_map, input);
  free(h_map);
} // end gather()


} // end dispatch

} // end detail

} // end komrade

