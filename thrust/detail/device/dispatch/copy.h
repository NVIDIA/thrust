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

#include <thrust/detail/type_traits.h>
#include <thrust/detail/device/omp/copy.h>
#include <thrust/detail/device/cuda/copy.h>

namespace thrust
{

namespace detail
{

namespace device
{

namespace dispatch
{


// omp path
// XXX this dispatch process is pretty lousy,
//     but we can't implement copy(host,omp) & copy(omp,host)
//     with generic::copy
template<typename InputIterator,
         typename OutputIterator>
  OutputIterator copy(InputIterator first,
                      InputIterator last,
                      OutputIterator result,
                      thrust::detail::false_type) // neither space is CUDA
{
  return thrust::detail::device::omp::copy(first, last, result);
} // end copy()


// at least one space is CUDA
template<typename InputIterator,
         typename OutputIterator>
  OutputIterator copy(InputIterator first,
                      InputIterator last,
                      OutputIterator result,
                      thrust::detail::true_type) // one of the spaces is CUDA
{
  return thrust::detail::device::cuda::copy(first, last, result);
} // end copy()


// entry point
template<typename InputIterator,
         typename OutputIterator,
         typename Space1,
         typename Space2>
  OutputIterator copy(InputIterator first,
                      InputIterator last,
                      OutputIterator result,
                      Space1,
                      Space2)
{
  // inspect both spaces
  typedef typename thrust::detail::integral_constant<bool,
    thrust::detail::is_convertible<Space1,thrust::detail::cuda_device_space_tag>::value ||
    thrust::detail::is_convertible<Space2,thrust::detail::cuda_device_space_tag>::value
  > is_one_of_the_spaces_cuda;

  return copy(first, last, result,
    is_one_of_the_spaces_cuda());
} // end copy()


} // end dispatch

} // end device

} // end detail

} // end thrust

