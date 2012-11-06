/*
 *  Copyright 2008-2012 NVIDIA Corporation
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

#include <thrust/detail/config.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/functional.h>
#include <thrust/system/cuda/detail/block/inclusive_scan.h>

namespace thrust
{
namespace system
{
namespace cuda
{
namespace detail
{
namespace block
{


template<typename Context, typename RandomAccessIterator, typename T, typename BinaryFunction>
inline __device__
typename thrust::iterator_value<RandomAccessIterator>::type
  inplace_exclusive_scan(Context &ctx, RandomAccessIterator first, T init, BinaryFunction op)
{
  // perform an inclusive scan, then shift right
  block::inplace_inclusive_scan(ctx, first, op);

  typename thrust::iterator_value<RandomAccessIterator>::type carry = first[ctx.block_dimension() - 1];

  ctx.barrier();

  typename thrust::iterator_value<RandomAccessIterator>::type left = (ctx.thread_index() == 0) ? init : first[ctx.thread_index() - 1];

  ctx.barrier();

  first[ctx.thread_index()] = left;

  ctx.barrier();

  return carry;
}


template<typename Context, typename Iterator, typename T>
inline __device__
  typename thrust::iterator_value<Iterator>::type
    inplace_exclusive_scan(Context &ctx, Iterator first, T init)
{
  return block::inplace_exclusive_scan(ctx, first, init, thrust::plus<typename thrust::iterator_value<Iterator>::type>());
}


} // end namespace block
} // end namespace detail
} // end namespace cuda
} // end namespace system
} // end namespace thrust

