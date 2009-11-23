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

#pragma once

#include <thrust/detail/device/dereference.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/distance.h>

namespace thrust
{
namespace detail
{
namespace device
{
namespace omp
{

namespace detail
{

template<typename InputIterator,
         typename OutputIterator>
OutputIterator copy(InputIterator first,
                    InputIterator last,
                    OutputIterator result,
                    thrust::detail::omp_device_space_tag,
                    thrust::detail::omp_device_space_tag)
{
  typedef typename thrust::iterator_difference<InputIterator>::type difference;
  difference n = thrust::distance(first,last);

#pragma omp parallel for
  for(difference i = 0;
      i < n;
      ++i)
  {
    dereference(result,i) = dereference(first,i);
  }

  return result + n;
} 


template<typename InputIterator,
         typename OutputIterator>
OutputIterator copy(InputIterator first,
                    InputIterator last,
                    OutputIterator result,
                    thrust::host_space_tag,
                    thrust::detail::omp_device_space_tag)
{
  typedef typename thrust::iterator_difference<InputIterator>::type difference;
  difference n = thrust::distance(first,last);

#pragma omp parallel for
  for(difference i = 0;
      i < n;
      ++i)
  {
    dereference(result,i) = first[i];
  }

  return result + n;
} 


template<typename InputIterator,
         typename OutputIterator>
OutputIterator copy(InputIterator first,
                    InputIterator last,
                    OutputIterator result,
                    thrust::detail::omp_device_space_tag,
                    thrust::host_space_tag)
{
  typedef typename thrust::iterator_difference<InputIterator>::type difference;
  difference n = thrust::distance(first,last);

#pragma omp parallel for
  for(difference i = 0;
      i < n;
      ++i)
  {
    result[i] = dereference(first,i);
  }

  return result + n;
} 

} // end detail

template<typename InputIterator,
         typename OutputIterator>
OutputIterator copy(InputIterator first,
                    InputIterator last,
                    OutputIterator result)
{
  // dispatch on space
  return thrust::detail::device::omp::detail::copy(first, last, result,
    typename thrust::iterator_space<InputIterator>::type(),
    typename thrust::iterator_space<OutputIterator>::type());
} 


} // end namespace omp
} // end namespace device
} // end namespace detail
} // end namespace thrust

