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
#include <thrust/pair.h>
#include <thrust/detail/device/omp/reduce.h>
#include <thrust/detail/device/cuda/reduce.h>
#include <thrust/iterator/detail/backend_iterator_categories.h>

namespace thrust
{
namespace detail
{
namespace device
{
namespace dispatch
{


template<typename RandomAccessIterator,
         typename SizeType,
         typename OutputType,
         typename BinaryFunction>
  SizeType get_unordered_blocked_reduce_n_schedule(RandomAccessIterator first,
                                                   SizeType n,
                                                   OutputType init,
                                                   BinaryFunction binary_op,
                                                   thrust::detail::omp_device_space_tag)
{
  // OpenMP implementation
  return thrust::detail::device::omp::get_unordered_blocked_reduce_n_schedule(first, n, init, binary_op);
}

template<typename RandomAccessIterator,
         typename SizeType,
         typename OutputType,
         typename BinaryFunction>
  SizeType get_unordered_blocked_reduce_n_schedule(RandomAccessIterator first,
                                                   SizeType n,
                                                   OutputType init,
                                                   BinaryFunction binary_op,
                                                   thrust::detail::cuda_device_space_tag)
{
  // CUDA implementation
  return thrust::detail::device::cuda::get_unordered_blocked_reduce_n_schedule(first, n, init, binary_op);
}

template<typename RandomAccessIterator,
         typename SizeType,
         typename OutputType,
         typename BinaryFunction>
  SizeType get_unordered_blocked_reduce_n_schedule(RandomAccessIterator first,
                                                   SizeType n,
                                                   OutputType init,
                                                   BinaryFunction binary_op,
                                                   thrust::any_space_tag)
{
  // default implementation
  return thrust::detail::device::dispatch::get_unordered_blocked_reduce_n_schedule(first, n, init, binary_op,
    thrust::detail::default_device_space_tag());
}


template<typename RandomAccessIterator1,
         typename SizeType1,
         typename SizeType2,
         typename BinaryFunction,
         typename RandomAccessIterator2>
  void unordered_blocked_reduce_n(RandomAccessIterator1 first,
                                  SizeType1 n,
                                  SizeType2 num_blocks,
                                  BinaryFunction binary_op,
                                  RandomAccessIterator2 result,
                                  thrust::detail::omp_device_space_tag)
{
  // OpenMP implementation
  return thrust::detail::device::omp::unordered_blocked_reduce_n(first, n, num_blocks, binary_op, result);
}

template<typename RandomAccessIterator1,
         typename SizeType1,
         typename SizeType2,
         typename BinaryFunction,
         typename RandomAccessIterator2>
  void unordered_blocked_reduce_n(RandomAccessIterator1 first,
                                  SizeType1 n,
                                  SizeType2 num_blocks,
                                  BinaryFunction binary_op,
                                  RandomAccessIterator2 result,
                                  thrust::detail::cuda_device_space_tag)
{
  // CUDA implementation
  return thrust::detail::device::cuda::unordered_blocked_reduce_n(first, n, num_blocks, binary_op, result);
}

template<typename RandomAccessIterator1,
         typename SizeType1,
         typename SizeType2,
         typename BinaryFunction,
         typename RandomAccessIterator2>
  void unordered_blocked_reduce_n(RandomAccessIterator1 first,
                                  SizeType1 n,
                                  SizeType2 num_blocks,
                                  BinaryFunction binary_op,
                                  RandomAccessIterator2 result,
                                  thrust::any_space_tag)
{
  // default implementation
  return thrust::detail::device::dispatch::unordered_blocked_reduce_n(first, n, num_blocks, binary_op, result,
    thrust::detail::default_device_space_tag());
}


} // end namespace dispatch
} // end namespace device
} // end namespace detail
} // end namespace thrust

