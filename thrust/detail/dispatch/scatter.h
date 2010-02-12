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


/*! \file scatter.h
 *  \brief Dispatch layer for scatter functions
 */

#pragma once

#include <thrust/iterator/iterator_traits.h>

#include <thrust/detail/host/scatter.h>
#include <thrust/detail/device/scatter.h>

#include <thrust/distance.h>
#include <thrust/detail/raw_buffer.h>

namespace thrust
{

namespace detail
{

// forward declarations
template<typename,typename> class raw_buffer;
template<typename>          class raw_host_buffer;

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
               thrust::host_space_tag,
               thrust::host_space_tag,
               thrust::host_space_tag)
{
    thrust::detail::host::scatter(first, last, map, output);
}


template<typename InputIterator1,
         typename InputIterator2,
         typename RandomAccessIterator>
  void scatter(InputIterator1 first,
               InputIterator1 last,
               InputIterator2 map,
               RandomAccessIterator output,
               thrust::any_space_tag,
               thrust::host_space_tag,
               thrust::host_space_tag)
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
                  thrust::host_space_tag,
                  thrust::host_space_tag,
                  thrust::host_space_tag,
                  thrust::host_space_tag)
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
               thrust::device_space_tag,
               thrust::device_space_tag,
               thrust::device_space_tag)
{
    thrust::detail::device::scatter(first, last, map, output);
}


template<typename InputIterator1,
         typename InputIterator2,
         typename RandomAccessIterator>
  void scatter(InputIterator1 first,
               InputIterator1 last,
               InputIterator2 map,
               RandomAccessIterator output,
               thrust::any_space_tag,
               thrust::device_space_tag,
               thrust::device_space_tag)
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
                  thrust::device_space_tag,
                  thrust::device_space_tag,
                  thrust::device_space_tag,
                  thrust::device_space_tag)
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
               thrust::device_space_tag,  // input on device
               thrust::host_space_tag,    // map on host
               thrust::host_space_tag)    // destination on host
{
  // copy input to host and scatter on host
  typedef typename thrust::iterator_traits<InputIterator1>::value_type InputType;
  thrust::detail::raw_host_buffer<InputType> buffer(first,last);
  thrust::scatter(buffer.begin(), buffer.end(), map, output);
} // end scatter()


template<typename InputIterator1,
         typename InputIterator2,
         typename RandomAccessIterator>
  void scatter(InputIterator1 first,
               InputIterator1 last,
               InputIterator2 map,
               RandomAccessIterator output,
               thrust::device_space_tag,  // input on device
               thrust::device_space_tag,  // map on device
               thrust::host_space_tag)    // destination on host
{
  // copy map to host and try again
  typedef typename thrust::iterator_traits<InputIterator2>::value_type IndexType;
  thrust::detail::raw_host_buffer<IndexType> h_map(map, map + thrust::distance(first,last));
  thrust::scatter(first, last, h_map.begin(), output);
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
               thrust::host_space_tag,    // input on host
               thrust::device_space_tag,  // map on device
               thrust::device_space_tag)  // destination on device
{
  // copy input to device and scatter on device
  typedef typename thrust::iterator_traits<InputIterator1>::value_type InputType;
  typedef typename thrust::iterator_space<RandomAccessIterator>::type Space;
  thrust::detail::raw_buffer<InputType,Space> buffer(first, last);
  thrust::scatter(buffer.begin(), buffer.end(), map, output);
} // end scatter()


template<typename InputIterator1,
         typename InputIterator2,
         typename RandomAccessIterator>
  void scatter(InputIterator1 first,
               InputIterator1 last,
               InputIterator2 map,
               RandomAccessIterator output,
               thrust::host_space_tag,    // input on host
               thrust::host_space_tag,    // map on host
               thrust::device_space_tag)  // destination on device
{
  // copy map to device and try again
  typedef typename thrust::iterator_traits<InputIterator2>::value_type IndexType;
  typedef typename thrust::iterator_space<RandomAccessIterator>::type Space;
  thrust::detail::raw_buffer<IndexType,Space> d_map(map, map + thrust::distance(first,last));
  thrust::scatter(first, last, d_map.begin(), output);
} // end scatter()

} // end namespace dispatch
} // end namespace detail
} // end namespace thrust

