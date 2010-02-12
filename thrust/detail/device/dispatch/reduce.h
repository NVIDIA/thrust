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

#pragma once

#include <thrust/iterator/iterator_traits.h>

#include <thrust/detail/device_ptr_category.h>

#include <thrust/detail/device/cuda/reduce.h>
#include <thrust/detail/device/omp/reduce.h>

namespace thrust
{
namespace detail
{
namespace device
{
namespace dispatch
{

template<typename InputIterator, 
         typename OutputType,
         typename BinaryFunction>
  OutputType reduce(InputIterator first,
                    InputIterator last,
                    OutputType init,
                    BinaryFunction binary_op,
                    thrust::detail::omp_device_space_tag)
{
    // OpenMP implementation
    return thrust::detail::device::omp::reduce(first, last, init, binary_op);
}

template<typename InputIterator, 
         typename OutputType,
         typename BinaryFunction>
  OutputType reduce(InputIterator first,
                    InputIterator last,
                    OutputType init,
                    BinaryFunction binary_op,
                    thrust::detail::cuda_device_space_tag)
{
    // CUDA implementation
    return thrust::detail::device::cuda::reduce(first, last, init, binary_op);
}

template<typename InputIterator, 
         typename OutputType,
         typename BinaryFunction>
  OutputType reduce(InputIterator first,
                    InputIterator last,
                    OutputType init,
                    BinaryFunction binary_op,
                    thrust::any_space_tag)
{
    // Use default backend
    return thrust::detail::device::dispatch::reduce(first, last, init, binary_op,
            thrust::detail::default_device_space_tag());
}

} // end namespace dispatch
} // end namespace device
} // end namespace detail
} // end namespace thrust

