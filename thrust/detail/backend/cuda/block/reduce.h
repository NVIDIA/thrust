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

// do not attempt to compile this file, which uses CUDA built-in variables, with any compiler other than nvcc
#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC

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

template <unsigned int block_size, typename ValueIterator, typename BinaryFunction>
__device__ __forceinline__
void reduce(ValueIterator data, BinaryFunction binary_op)
{
  // TODO generalize this code with TMP
  if (block_size >= 1024) { if (threadIdx.x < 512) { data[threadIdx.x] = binary_op(data[threadIdx.x], data[threadIdx.x + 512]); } __syncthreads(); }
  if (block_size >=  512) { if (threadIdx.x < 256) { data[threadIdx.x] = binary_op(data[threadIdx.x], data[threadIdx.x + 256]); } __syncthreads(); }
  if (block_size >=  256) { if (threadIdx.x < 128) { data[threadIdx.x] = binary_op(data[threadIdx.x], data[threadIdx.x + 128]); } __syncthreads(); }
  if (block_size >=  128) { if (threadIdx.x <  64) { data[threadIdx.x] = binary_op(data[threadIdx.x], data[threadIdx.x +  64]); } __syncthreads(); }
  if (block_size >=   64) { if (threadIdx.x <  32) { data[threadIdx.x] = binary_op(data[threadIdx.x], data[threadIdx.x +  32]); } __syncthreads(); }
  if (block_size >=   32) { if (threadIdx.x <  16) { data[threadIdx.x] = binary_op(data[threadIdx.x], data[threadIdx.x +  16]); } __syncthreads(); }
  if (block_size >=   16) { if (threadIdx.x <   8) { data[threadIdx.x] = binary_op(data[threadIdx.x], data[threadIdx.x +   8]); } __syncthreads(); }
  if (block_size >=    8) { if (threadIdx.x <   4) { data[threadIdx.x] = binary_op(data[threadIdx.x], data[threadIdx.x +   4]); } __syncthreads(); }
  if (block_size >=    4) { if (threadIdx.x <   2) { data[threadIdx.x] = binary_op(data[threadIdx.x], data[threadIdx.x +   2]); } __syncthreads(); }
  if (block_size >=    2) { if (threadIdx.x <   1) { data[threadIdx.x] = binary_op(data[threadIdx.x], data[threadIdx.x +   1]); } __syncthreads(); }
}

template <typename ValueIterator, typename BinaryFunction>
__device__ __forceinline__
void reduce_n(ValueIterator data, unsigned int n, BinaryFunction binary_op)
{
  if (blockDim.x < n)
  {
    for (unsigned int i = blockDim.x + threadIdx.x; i < n; i += blockDim.x)
      data[threadIdx.x] = binary_op(data[threadIdx.x], data[i]);

    __syncthreads();
  }

  while (n > 1)
  {
    unsigned int half = n / 2;

    if (threadIdx.x < half)
      data[threadIdx.x] = binary_op(data[threadIdx.x], data[n - threadIdx.x - 1]);

    __syncthreads();

    n = n - half;
  }
}

} // end namespace block
} // end namespace cuda
} // end namespace backend
} // end namespace detail
} // end namespace thrust

#endif // THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC

