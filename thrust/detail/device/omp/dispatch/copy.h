/*
 *  Copyright 2008-2011 NVIDIA Corporation
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

#pragma once

#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/detail/minimum_category.h>
#include <thrust/detail/device/omp/copy_device_to_device.h>
#include <thrust/detail/device/omp/copy_host_or_any_to_device.h>
#include <thrust/detail/device/omp/copy_device_to_host_or_any.h>

// for std::copy
#include <algorithm>

namespace thrust
{

namespace detail
{

namespace device
{

namespace omp
{

namespace dispatch
{


namespace detail
{

// TODO eliminate these three specializations when we no longer need device::dereference()
// device to device
template<typename InputIterator,
         typename OutputIterator>
OutputIterator copy(InputIterator first,
                    InputIterator last,
                    OutputIterator result,
                    thrust::detail::omp_device_space_tag,
                    thrust::detail::omp_device_space_tag)
{
  return thrust::detail::device::omp::copy_device_to_device(first,last,result);
}


// host or any to device
template<typename InputIterator,
         typename OutputIterator,
         typename HostOrAnySpaceTag>
OutputIterator copy(InputIterator first,
                    InputIterator last,
                    OutputIterator result,
                    HostOrAnySpaceTag,
                    thrust::detail::omp_device_space_tag)
{
  return thrust::detail::device::omp::copy_host_or_any_to_device(first,last,result);
}


// device to host or any
template<typename InputIterator,
         typename OutputIterator,
         typename HostOrAnySpaceTag>
OutputIterator copy(InputIterator first,
                    InputIterator last,
                    OutputIterator result,
                    thrust::detail::omp_device_space_tag,
                    HostOrAnySpaceTag)
{
  return thrust::detail::device::omp::copy_device_to_host_or_any(first,last,result);
}

} // end detail

// random access to random access
template<typename InputIterator,
         typename OutputIterator>
OutputIterator copy(InputIterator first,
                    InputIterator last,
                    OutputIterator result,
                    thrust::random_access_traversal_tag)
{
  // dispatch on space
  return thrust::detail::device::omp::dispatch::detail::copy(first, last, result,
    typename thrust::iterator_space<InputIterator>::type(),
    typename thrust::iterator_space<OutputIterator>::type());
} 

template<typename InputIterator,
         typename Size,
         typename OutputIterator>
OutputIterator copy_n(InputIterator first,
                      Size n,
                      OutputIterator result,
                      thrust::random_access_traversal_tag)
{
  // implement with copy
  return thrust::detail::device::omp::dispatch::copy(first, first + n, result,
    thrust::random_access_traversal_tag());
} 

// incrementable to incrementable
template<typename InputIterator,
         typename OutputIterator>
OutputIterator copy(InputIterator first,
                    InputIterator last,
                    OutputIterator result,
                    thrust::incrementable_traversal_tag)
{
  // serialize on the host
  return std::copy(first,last,result);
}

template<typename InputIterator,
         typename Size,
         typename OutputIterator>
OutputIterator copy_n(InputIterator first,
                      Size n,
                      OutputIterator result,
                      thrust::incrementable_traversal_tag)
{
  // serialize on the host
  for(; n > Size(0); ++first, ++result, --n)
    *result = *first;
  return result;
}

} // end dispatch

} // end omp

} // end device

} // end detail

} // end thrust

