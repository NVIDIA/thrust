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
#include <thrust/detail/device/cuda/for_each.h>
#include <thrust/detail/device/omp/for_each.h>

namespace thrust
{

namespace detail
{

namespace device
{

namespace dispatch
{

template<typename RandomAccessIterator,
         typename Size,
         typename UnaryFunction>
  RandomAccessIterator for_each_n(RandomAccessIterator first,
                                  Size n,
                                  UnaryFunction f,
                                  thrust::detail::omp_device_space_tag)
{
  // OpenMP implementation
  return thrust::detail::device::omp::for_each_n(first, n, f);
}

template<typename RandomAccessIterator,
         typename Size,
         typename UnaryFunction>
  RandomAccessIterator for_each_n(RandomAccessIterator first,
                                  Size n,
                                  UnaryFunction f,
                                  thrust::detail::cuda_device_space_tag)
{
  // CUDA implementation
  return thrust::detail::device::cuda::for_each_n(first, n, f);
}

template<typename RandomAccessIterator,
         typename Size,
         typename UnaryFunction>
  RandomAccessIterator for_each_n(RandomAccessIterator first,
                                  Size n,
                                  UnaryFunction f,
                                  thrust::any_space_tag)
{
  // default implementation
  return thrust::detail::device::dispatch::for_each_n(first, n, f,
    thrust::detail::default_device_space_tag());
}

} // end dispatch

} // end device

} // end detail

} // end thrust

