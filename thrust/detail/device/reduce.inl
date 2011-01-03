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

#include <thrust/detail/device/dispatch/reduce.h>
#include <thrust/detail/device/generic/reduce_by_key.h>
#include <thrust/detail/device/generic/reduce.h>

#include <thrust/iterator/iterator_traits.h>

namespace thrust
{
namespace detail
{
namespace device
{


template<typename InputIterator, 
         typename OutputType,
         typename BinaryFunction>
  OutputType reduce(InputIterator first,
                    InputIterator last,
                    OutputType init,
                    BinaryFunction binary_op)
{
  // use generic path
  return thrust::detail::device::generic::reduce_n(first, last - first, init, binary_op);
}

template <typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator1,
          typename OutputIterator2,
          typename BinaryPredicate,
          typename BinaryFunction>
  thrust::pair<OutputIterator1,OutputIterator2>
  reduce_by_key(InputIterator1 keys_first, 
                     InputIterator1 keys_last,
                     InputIterator2 values_first,
                     OutputIterator1 keys_output,
                     OutputIterator2 values_output,
                     BinaryPredicate binary_pred,
                     BinaryFunction binary_op)
{
  // use generic path
  return thrust::detail::device::generic::reduce_by_key(keys_first, keys_last, values_first, keys_output, values_output, binary_pred, binary_op);
}

template<typename RandomAccessIterator,
         typename SizeType,
         typename OutputType,
         typename BinaryFunction>
  SizeType get_unordered_blocked_reduce_n_schedule(RandomAccessIterator first,
                                                   SizeType n,
                                                   OutputType init,
                                                   BinaryFunction binary_op)
{
  return thrust::detail::device::dispatch::get_unordered_blocked_reduce_n_schedule(first, n, init, binary_op,
    typename thrust::iterator_space<RandomAccessIterator>::type());
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
                                  RandomAccessIterator2 result)
{
  return thrust::detail::device::dispatch::unordered_blocked_reduce_n(first, n, num_blocks, binary_op, result,
    typename thrust::iterator_space<RandomAccessIterator1>::type());
}


} // end device
} // end detail
} // end thrust

