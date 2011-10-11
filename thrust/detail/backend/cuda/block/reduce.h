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

#include <thrust/detail/config.h>

#include <thrust/iterator/iterator_traits.h>

namespace thrust
{
namespace detail
{
namespace backend
{
namespace cuda
{
namespace block
{

template <typename Context, unsigned int block_size, typename ValueIterator, typename BinaryFunction>
__device__ __forceinline__
void reduce(Context context, ValueIterator data, BinaryFunction binary_op)
{
  // TODO generalize this code with TMP
  if (block_size >= 1024) { if (context.thread_index() < 512) { data[context.thread_index()] = binary_op(data[context.thread_index()], data[context.thread_index() + 512]); } context.barrier(); }
  if (block_size >=  512) { if (context.thread_index() < 256) { data[context.thread_index()] = binary_op(data[context.thread_index()], data[context.thread_index() + 256]); } context.barrier(); }
  if (block_size >=  256) { if (context.thread_index() < 128) { data[context.thread_index()] = binary_op(data[context.thread_index()], data[context.thread_index() + 128]); } context.barrier(); }
  if (block_size >=  128) { if (context.thread_index() <  64) { data[context.thread_index()] = binary_op(data[context.thread_index()], data[context.thread_index() +  64]); } context.barrier(); }
  if (block_size >=   64) { if (context.thread_index() <  32) { data[context.thread_index()] = binary_op(data[context.thread_index()], data[context.thread_index() +  32]); } context.barrier(); }
  if (block_size >=   32) { if (context.thread_index() <  16) { data[context.thread_index()] = binary_op(data[context.thread_index()], data[context.thread_index() +  16]); } context.barrier(); }
  if (block_size >=   16) { if (context.thread_index() <   8) { data[context.thread_index()] = binary_op(data[context.thread_index()], data[context.thread_index() +   8]); } context.barrier(); }
  if (block_size >=    8) { if (context.thread_index() <   4) { data[context.thread_index()] = binary_op(data[context.thread_index()], data[context.thread_index() +   4]); } context.barrier(); }
  if (block_size >=    4) { if (context.thread_index() <   2) { data[context.thread_index()] = binary_op(data[context.thread_index()], data[context.thread_index() +   2]); } context.barrier(); }
  if (block_size >=    2) { if (context.thread_index() <   1) { data[context.thread_index()] = binary_op(data[context.thread_index()], data[context.thread_index() +   1]); } context.barrier(); }
}

template <typename Context, typename ValueIterator, typename BinaryFunction>
__device__ __forceinline__
void reduce_n(Context context, ValueIterator data, unsigned int n, BinaryFunction binary_op)
{
  if (context.block_dimension() < n)
  {
    for (unsigned int i = context.block_dimension() + context.thread_index(); i < n; i += context.block_dimension())
      data[context.thread_index()] = binary_op(data[context.thread_index()], data[i]);

    context.barrier();
  }

  while (n > 1)
  {
    unsigned int half = n / 2;

    if (context.thread_index() < half)
      data[context.thread_index()] = binary_op(data[context.thread_index()], data[n - context.thread_index() - 1]);

    context.barrier();

    n = n - half;
  }
}

} // end namespace block
} // end namespace cuda
} // end namespace backend
} // end namespace detail
} // end namespace thrust

