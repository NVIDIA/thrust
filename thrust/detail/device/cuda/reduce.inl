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


/*! \file reduce.inl
 *  \brief Inline file for reduce.h
 */

#pragma once

#include <thrust/functional.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/detail/static_assert.h>
#include <thrust/distance.h>
#include <thrust/detail/raw_buffer.h>

#include <thrust/detail/device/cuda/reduce_n.h>

namespace thrust
{
namespace detail
{
namespace device
{
namespace cuda
{

template<typename InputIterator, 
         typename OutputType,
         typename BinaryFunction>
  OutputType reduce(InputIterator first,
                    InputIterator last,
                    OutputType init,
                    BinaryFunction binary_op)
{
  // we're attempting to launch a kernel, assert we're compiling with nvcc
  // ========================================================================
  // X Note to the user: If you've found this line due to a compiler error, X
  // X you need to compile your code using nvcc, rather than g++ or cl.exe  X
  // ========================================================================
  THRUST_STATIC_ASSERT( (depend_on_instantiation<InputIterator, THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC>::value) );

  // check for empty range
  if(first == last) return init;

  typedef typename thrust::iterator_difference<InputIterator>::type Size;
  const Size n = thrust::distance(first,last);

  // compute schedule for first stage
  const Size num_blocks = thrust::detail::device::cuda::get_unordered_blocked_reduce_n_schedule(first, n, init, binary_op);

  // allocate storage for the initializer and partial sums
  thrust::detail::raw_cuda_device_buffer<OutputType> partial_sums(1 + num_blocks);

  // set first element of temp array to init
  partial_sums[0] = init;

  // accumulate partial sums
  thrust::detail::device::cuda::unordered_blocked_reduce_n(first, n, num_blocks, binary_op, partial_sums.begin() + 1);

  // reduce partial sums
  thrust::detail::device::cuda::unordered_blocked_reduce_n(partial_sums.begin(), 1 + num_blocks, 1, binary_op, partial_sums.begin());

  return partial_sums[0];
}

} // end namespace cuda
} // end namespace device
} // end namespace detail
} // end namespace thrust

