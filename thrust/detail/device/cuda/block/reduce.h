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

// do not attempt to compile this file, which uses CUDA built-in variables, with any compiler other than nvcc
#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC

namespace thrust
{
namespace detail
{
namespace device
{
namespace cuda
{
namespace block
{

template <unsigned int block_size, typename ValueIterator, typename BinaryFunction>
__device__ 
void reduce(ValueIterator data, BinaryFunction binary_op)
{
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
__device__ 
void reduce_n(ValueIterator data, const unsigned int n, BinaryFunction binary_op)
{
    // assume n <= 1024
    if (n > 512) { if (threadIdx.x < 512 && threadIdx.x + 512 < n) { data[threadIdx.x] = binary_op(data[threadIdx.x], data[threadIdx.x + 512]); } __syncthreads(); }
    if (n > 256) { if (threadIdx.x < 256 && threadIdx.x + 256 < n) { data[threadIdx.x] = binary_op(data[threadIdx.x], data[threadIdx.x + 256]); } __syncthreads(); }
    if (n > 128) { if (threadIdx.x < 128 && threadIdx.x + 128 < n) { data[threadIdx.x] = binary_op(data[threadIdx.x], data[threadIdx.x + 128]); } __syncthreads(); }
    if (n >  64) { if (threadIdx.x <  64 && threadIdx.x +  64 < n) { data[threadIdx.x] = binary_op(data[threadIdx.x], data[threadIdx.x +  64]); } __syncthreads(); }
    if (n >  32) { if (threadIdx.x <  32 && threadIdx.x +  32 < n) { data[threadIdx.x] = binary_op(data[threadIdx.x], data[threadIdx.x +  32]); } __syncthreads(); }
    if (n >  16) { if (threadIdx.x <  16 && threadIdx.x +  16 < n) { data[threadIdx.x] = binary_op(data[threadIdx.x], data[threadIdx.x +  16]); } __syncthreads(); }
    if (n >   8) { if (threadIdx.x <   8 && threadIdx.x +   8 < n) { data[threadIdx.x] = binary_op(data[threadIdx.x], data[threadIdx.x +   8]); } __syncthreads(); }
    if (n >   4) { if (threadIdx.x <   4 && threadIdx.x +   4 < n) { data[threadIdx.x] = binary_op(data[threadIdx.x], data[threadIdx.x +   4]); } __syncthreads(); }
    if (n >   2) { if (threadIdx.x <   2 && threadIdx.x +   2 < n) { data[threadIdx.x] = binary_op(data[threadIdx.x], data[threadIdx.x +   2]); } __syncthreads(); }
    if (n >   1) { if (threadIdx.x <   1 && threadIdx.x +   1 < n) { data[threadIdx.x] = binary_op(data[threadIdx.x], data[threadIdx.x +   1]); } __syncthreads(); }
}

} // end namespace block
} // end namespace cuda
} // end namespace device
} // end namespace detail
} // end namespace thrust

#endif // THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC

