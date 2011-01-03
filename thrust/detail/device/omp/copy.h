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
#include <thrust/detail/device/omp/dispatch/copy.h>

namespace thrust
{
namespace detail
{
namespace device
{
namespace omp
{


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
  return thrust::detail::device::omp::dispatch::copy(first, last, result, minimum_traversal());
} 

template<typename InputIterator,
         typename Size,
         typename OutputIterator>
OutputIterator copy_n(InputIterator first,
                      Size n,
                      OutputIterator result)
{
  typedef typename thrust::iterator_traversal<InputIterator>::type traversal1;
  typedef typename thrust::iterator_traversal<OutputIterator>::type traversal2;

  // XXX minimum_traversal doesn't exist, but it should?
  //typedef typename thrust::detail::minimum_traversal<traversal1,traversal2>::type minimum_traversal;
  typedef typename thrust::detail::minimum_category<traversal1,traversal2>::type minimum_traversal;

  // dispatch on min traversal
  return thrust::detail::device::omp::dispatch::copy_n(first, n, result, minimum_traversal());
} 


} // end namespace omp
} // end namespace device
} // end namespace detail
} // end namespace thrust

