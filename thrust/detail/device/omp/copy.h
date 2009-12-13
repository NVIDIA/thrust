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
#include <thrust/iterator/detail/minimum_category.h>

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

namespace detail
{

// XXX eliminate these 3 overloads
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
    InputIterator  first_temp  = first  + i;
    OutputIterator result_temp = result + i;

    dereference(result_temp) = dereference(first_temp);
  }

  return result + n;
} 


template<typename InputIterator,
         typename OutputIterator,
         typename HostOrAnySpaceTag>
OutputIterator copy(InputIterator first,
                    InputIterator last,
                    OutputIterator result,
                    HostOrAnySpaceTag,
                    thrust::detail::omp_device_space_tag)
{
  typedef typename thrust::iterator_difference<InputIterator>::type difference;
  difference n = thrust::distance(first,last);

#pragma omp parallel for
  for(difference i = 0;
      i < n;
      ++i)
  {
    OutputIterator temp = result + i;
    dereference(temp) = first[i];
  }

  return result + n;
} 


template<typename InputIterator,
         typename OutputIterator,
         typename HostOrAnySpaceTag>
OutputIterator copy(InputIterator first,
                    InputIterator last,
                    OutputIterator result,
                    thrust::detail::omp_device_space_tag,
                    HostOrAnySpaceTag)
{
  typedef typename thrust::iterator_difference<InputIterator>::type difference;
  difference n = thrust::distance(first,last);

#pragma omp parallel for
  for(difference i = 0;
      i < n;
      ++i)
  {
    InputIterator temp = first + i;
    result[i] = dereference(temp);
  }

  return result + n;
} 


template<typename InputIterator,
         typename OutputIterator>
OutputIterator copy(InputIterator first,
                    InputIterator last,
                    OutputIterator result,
                    thrust::random_access_traversal_tag)
{
  // dispatch on space
  return thrust::detail::device::omp::detail::copy(first, last, result,
    typename thrust::iterator_space<InputIterator>::type(),
    typename thrust::iterator_space<OutputIterator>::type());
} 


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


} // end detail


// entry point
template<typename InputIterator,
         typename OutputIterator>
OutputIterator copy(InputIterator first,
                    InputIterator last,
                    OutputIterator result)
{
  typedef typename thrust::iterator_traversal<InputIterator>::type traversal1;
  typedef typename thrust::iterator_traversal<OutputIterator>::type traversal2;

  // XXX minimum_traversal doesn't exist, but it should?
  //typedef typename thrust::detail::minimum_traversal<traversal1,traversal2>::type minimum_traversal;
  typedef typename thrust::detail::minimum_category<traversal1,traversal2>::type minimum_traversal;

  // dispatch on min traversal
  return thrust::detail::device::omp::detail::copy(first, last, result, minimum_traversal());
} 


} // end namespace omp
} // end namespace device
} // end namespace detail
} // end namespace thrust

