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

#include <thrust/detail/device/cuda/scan.h>
#include <thrust/detail/device/omp/scan.h>

namespace thrust
{
namespace detail
{
namespace device
{
namespace dispatch
{

////////////////////////////
// OpenMP implementations //
////////////////////////////

template<typename InputIterator,
         typename OutputIterator,
         typename AssociativeOperator>
  OutputIterator inclusive_scan(InputIterator first,
                                InputIterator last,
                                OutputIterator result,
                                AssociativeOperator binary_op,
                                thrust::detail::omp_device_space_tag,
                                thrust::detail::omp_device_space_tag)
{
    return thrust::detail::device::omp::inclusive_scan(first, last, result, binary_op);
}

template<typename InputIterator,
         typename OutputIterator,
         typename T,
         typename AssociativeOperator>
  OutputIterator exclusive_scan(InputIterator first,
                                InputIterator last,
                                OutputIterator result,
                                T init,
                                AssociativeOperator binary_op,
                                thrust::detail::omp_device_space_tag,
                                thrust::detail::omp_device_space_tag)
{
    return thrust::detail::device::omp::exclusive_scan(first, last, result, init, binary_op);
}


//////////////////////////
// CUDA implementations //
//////////////////////////

template<typename InputIterator,
         typename OutputIterator,
         typename AssociativeOperator>
  OutputIterator inclusive_scan(InputIterator first,
                                InputIterator last,
                                OutputIterator result,
                                AssociativeOperator binary_op,
                                thrust::detail::cuda_device_space_tag,
                                thrust::detail::cuda_device_space_tag)
{
    return thrust::detail::device::cuda::inclusive_scan(first, last, result, binary_op);
}

template<typename InputIterator,
         typename OutputIterator,
         typename T,
         typename AssociativeOperator>
  OutputIterator exclusive_scan(InputIterator first,
                                InputIterator last,
                                OutputIterator result,
                                T init,
                                AssociativeOperator binary_op,
                                thrust::detail::cuda_device_space_tag,
                                thrust::detail::cuda_device_space_tag)
{
    return thrust::detail::device::cuda::exclusive_scan(first, last, result, init, binary_op);
}

} // end namespace dispatch
} // end namespace device
} // end namespace detail
} // end namespace thrust

