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

#include <thrust/detail/type_traits.h>
#include <thrust/detail/device/omp/copy.h>
#include <thrust/detail/device/cuda/copy.h>
#include <thrust/detail/device/generic/copy_if.h>
#include <thrust/detail/device/cuda/copy_if.h>

namespace thrust
{
namespace detail
{
namespace device
{
namespace dispatch
{

//////////////////
// OpenMP Paths //
//////////////////

// omp path
// XXX this dispatch process is pretty lousy,
//     but we can't implement copy(host,omp) & copy(omp,host)
//     with generic::copy
template<typename InputIterator,
         typename OutputIterator>
  OutputIterator copy(InputIterator first,
                      InputIterator last,
                      OutputIterator result,
                      thrust::detail::false_type) // no space is CUDA
{
  return thrust::detail::device::omp::copy(first, last, result);
} // end copy()

template<typename InputIterator,
         typename Size,
         typename OutputIterator>
  OutputIterator copy_n(InputIterator first,
                        Size n,
                        OutputIterator result,
                        thrust::detail::false_type) // no space is CUDA
{
  return thrust::detail::device::omp::copy_n(first, n, result);
} // end copy_n()

template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename Predicate>
  OutputIterator copy_if(InputIterator1 first,
                         InputIterator1 last,
                         InputIterator2 stencil,
                         OutputIterator result,
                         Predicate pred,
                         thrust::detail::false_type) // no space is CUDA
{
  return thrust::detail::device::generic::copy_if(first, last, stencil, result, pred);
} // end copy_if()

////////////////
// CUDA Paths //
////////////////

template<typename InputIterator,
         typename OutputIterator>
  OutputIterator copy(InputIterator first,
                      InputIterator last,
                      OutputIterator result,
                      thrust::detail::true_type) // at least one of the spaces is CUDA
{
  return thrust::detail::device::cuda::copy(first, last, result);
} // end copy()

template<typename InputIterator,
         typename Size,
         typename OutputIterator>
  OutputIterator copy_n(InputIterator first,
                        Size n,
                        OutputIterator result,
                        thrust::detail::true_type) // at least one of the spaces is CUDA
{
  return thrust::detail::device::cuda::copy_n(first, n, result);
} // end copy_n()

template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename Predicate>
  OutputIterator copy_if(InputIterator1 first,
                         InputIterator1 last,
                         InputIterator2 stencil,
                         OutputIterator result,
                         Predicate pred,
                         thrust::detail::true_type) // at least one of the spaces is CUDA
{
  return thrust::detail::device::cuda::copy_if(first, last, stencil, result, pred);
} // end copy_if()

//////////////////
// Entry points //
//////////////////

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
    thrust::detail::is_same<Space1,thrust::detail::cuda_device_space_tag>::value ||
    thrust::detail::is_same<Space2,thrust::detail::cuda_device_space_tag>::value
  > is_one_of_the_spaces_cuda;

  return copy(first, last, result,
    is_one_of_the_spaces_cuda());
} // end copy()

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
  // inspect both spaces
  typedef typename thrust::detail::integral_constant<bool,
    thrust::detail::is_same<Space1,thrust::detail::cuda_device_space_tag>::value ||
    thrust::detail::is_same<Space2,thrust::detail::cuda_device_space_tag>::value
  > is_one_of_the_spaces_cuda;

  return copy_n(first, n, result,
    is_one_of_the_spaces_cuda());
} // end copy_n()

template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename Predicate,
         typename Space1,
         typename Space2,
         typename Space3>
   OutputIterator copy_if(InputIterator1 first,
                          InputIterator1 last,
                          InputIterator2 stencil,
                          OutputIterator output,
                          Predicate pred,
                          Space1,
                          Space2,
                          Space3)
{
  // inspect all spaces
  typedef typename thrust::detail::integral_constant<bool,
    thrust::detail::is_same<Space1,thrust::detail::cuda_device_space_tag>::value ||
    thrust::detail::is_same<Space2,thrust::detail::cuda_device_space_tag>::value ||
    thrust::detail::is_same<Space3,thrust::detail::cuda_device_space_tag>::value
  > is_one_of_the_spaces_cuda;

  return copy_if(first, last, stencil, output, pred,
    is_one_of_the_spaces_cuda());
} // end copy_if()

} // end namespace dispatch
} // end namespace device
} // end namespace detail
} // end namespace thrust

