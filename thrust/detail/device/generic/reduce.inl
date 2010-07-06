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

#include <thrust/detail/device/generic/reduce.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/pair.h>
#include <thrust/detail/device/reduce.h>
#include <thrust/detail/raw_buffer.h>

namespace thrust
{

namespace detail
{

namespace device
{

namespace generic
{

template<typename RandomAccessIterator,
         typename SizeType,
         typename OutputType,
         typename BinaryFunction>
  OutputType reduce_n(RandomAccessIterator first,
                      SizeType n,
                      OutputType init,
                      BinaryFunction binary_op)
{
  // compute schedule for first stage
  const thrust::pair<SizeType,SizeType> blocking =
    thrust::detail::device::get_unordered_blocked_reduce_n_schedule(first, n, init, binary_op);

  const SizeType num_blocks = blocking.first;
  
  // allocate storage for the initializer and partial sums
  typedef typename thrust::iterator_space<RandomAccessIterator>::type Space;
  thrust::detail::raw_device_buffer<OutputType,Space> partial_sums(1 + num_blocks);
  
  // set first element of temp array to init
  partial_sums[0] = init;
  
  // accumulate partial sums
  thrust::detail::device::unordered_blocked_reduce_n(first, n, num_blocks, binary_op, partial_sums.begin() + 1);

  // reduce partial sums
  thrust::detail::device::unordered_blocked_reduce_n(partial_sums.begin(), num_blocks + 1, 1, binary_op, partial_sums.begin());

  return partial_sums[0];
} // end reduce_n()

} // end generic

} // end device

} // end detail

} // end thrust

