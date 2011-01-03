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
#include <thrust/iterator/detail/minimum_space.h>
#include <thrust/detail/type_traits.h>
#include <thrust/detail/device/cuda/copy_cross_space.h>
#include <thrust/detail/device/cuda/copy_device_to_device.h>

namespace thrust
{

namespace detail
{

namespace device
{

namespace cuda
{

namespace dispatch
{


///////////////////////
// CUDA to CUDA Path //
///////////////////////

template<typename InputIterator,
         typename OutputIterator>
  OutputIterator copy(InputIterator begin,
                      InputIterator end,
                      OutputIterator result,
                      thrust::detail::cuda_device_space_tag)
{
    return thrust::detail::device::cuda::copy_device_to_device(begin, end, result);
}

template<typename InputIterator,
         typename Size,
         typename OutputIterator>
  OutputIterator copy_n(InputIterator first,
                        Size n,
                        OutputIterator result,
                        thrust::detail::cuda_device_space_tag)
{
    return thrust::detail::device::cuda::copy_device_to_device(first, first + n, result);
}


///////////////
// Any Paths //
///////////////

template<typename InputIterator,
         typename OutputIterator>
  OutputIterator copy(InputIterator begin,
                      InputIterator end,
                      OutputIterator result,
                      thrust::any_space_tag)
{
    return thrust::detail::device::cuda::copy_device_to_device(begin, end, result);
}

template<typename InputIterator,
         typename Size,
         typename OutputIterator>
  OutputIterator copy_n(InputIterator first,
                        Size n,
                        OutputIterator result,
                        thrust::any_space_tag)
{
    return thrust::detail::device::cuda::copy_device_to_device(first, first + n, result);
}


//////////////////////
// Cross-Space Path //
//////////////////////

template<typename InputIterator,
         typename OutputIterator>
  OutputIterator copy(InputIterator first,
                      InputIterator last,
                      OutputIterator result,
                      thrust::detail::false_type intra_space_copy)
{
  return thrust::detail::device::cuda::copy_cross_space(first, last, result);
}

template<typename InputIterator,
         typename Size,
         typename OutputIterator>
  OutputIterator copy_n(InputIterator first,
                        Size n,
                        OutputIterator result,
                        thrust::detail::false_type intra_space_copy)
{
  return thrust::detail::device::cuda::copy_cross_space_n(first, n, result);
}


//////////////////////
// Intra-Space Path //
//////////////////////

template<typename InputIterator,
         typename OutputIterator>
  OutputIterator copy(InputIterator first,
                      InputIterator last,
                      OutputIterator result,
                      thrust::detail::true_type intra_space_copy)
{
  typedef typename thrust::iterator_space<InputIterator>::type  space1;
  typedef typename thrust::iterator_space<OutputIterator>::type space2;

  // find the minimum space of the two
  typedef typename thrust::detail::minimum_space<space1,space2>::type minimum_space;

  return copy(first, last, result, minimum_space());
}

template<typename InputIterator,
         typename Size,
         typename OutputIterator>
  OutputIterator copy_n(InputIterator first,
                        Size n,
                        OutputIterator result,
                        thrust::detail::true_type intra_space_copy)
{
  typedef typename thrust::iterator_space<InputIterator>::type  space1;
  typedef typename thrust::iterator_space<OutputIterator>::type space2;

  // find the minimum space of the two
  typedef typename thrust::detail::minimum_space<space1,space2>::type minimum_space;

  return copy_n(first, n, result, minimum_space());
}


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
  return copy(first, last, result,
    typename thrust::detail::is_one_convertible_to_the_other<Space1,Space2>::type());
}

template<typename InputIterator,
         typename Size,
         typename OutputIterator,
         typename Space1,
         typename Space2>
  OutputIterator copy_n(InputIterator first,
                        Size n,
                        OutputIterator result,
                        Space1,
                        Space2)
{
  return copy_n(first, n, result,
    typename thrust::detail::is_one_convertible_to_the_other<Space1,Space2>::type());
}

} // end dispatch

} // end cuda

} // end device

} // end detail

} // end thrust

