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

/* Reduces [data, data + n) using binary_op and stores the result in data[0]
 *
 * Upon return the elements in [data + 1, data + n) have unspecified values.
 */
template <typename Context, typename ValueIterator, typename BinaryFunction>
__device__ __thrust_forceinline__
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
} // end namespace detail
} // end namespace cuda
} // end namespace system
} // end namespace thrust

