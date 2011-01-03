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

#include <thrust/detail/device/cuda/dispatch/reduce.h>
#include <thrust/detail/device/cuda/reduce.h>

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

template<typename RandomAccessIterator,
         typename SizeType,
         typename OutputType,
         typename BinaryFunction>
  SizeType get_unordered_blocked_reduce_n_schedule(RandomAccessIterator first,
                                                   SizeType n,
                                                   OutputType init,
                                                   BinaryFunction binary_op,
                                                   thrust::detail::true_type)
{
  // wide reduction
  return thrust::detail::device::cuda::get_unordered_blocked_wide_reduce_n_schedule(first,n,init,binary_op);
}


template<typename RandomAccessIterator,
         typename SizeType,
         typename OutputType,
         typename BinaryFunction>
  SizeType get_unordered_blocked_reduce_n_schedule(RandomAccessIterator first,
                                                   SizeType n,
                                                   OutputType init,
                                                   BinaryFunction binary_op,
                                                   thrust::detail::false_type)
{
  // standard reduction
  return thrust::detail::device::cuda::get_unordered_blocked_standard_reduce_n_schedule(first,n,init,binary_op);
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
                                  thrust::detail::true_type)  // use wide reduction
{
  return thrust::detail::device::cuda::unordered_blocked_wide_reduce_n(first,n,num_blocks,binary_op,result);
} // end unordered_blocked_reduce_n()


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
                                  thrust::detail::false_type)  // use standard reduction
{
  return thrust::detail::device::cuda::unordered_blocked_standard_reduce_n(first,n,num_blocks,binary_op,result);
} // end unordered_blocked_reduce_n()


} // end dispatch
} // end cuda
} // end device
} // end detail
} // end thrust

