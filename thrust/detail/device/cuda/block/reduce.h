/*
 *  Copyright 2008-2009 NVIDIA Corporation
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

namespace thrust
{

namespace detail
{

namespace block
{

template <unsigned int BLOCK_SIZE, typename ValueIterator, typename BinaryFunction>
__device__ 
void reduce(ValueIterator data, BinaryFunction binary_op)
{
    if (BLOCK_SIZE >= 512) { if (threadIdx.x < 256) { data[threadIdx.x] = binary_op(data[threadIdx.x], data[threadIdx.x + 256]); } __syncthreads(); }
    if (BLOCK_SIZE >= 256) { if (threadIdx.x < 128) { data[threadIdx.x] = binary_op(data[threadIdx.x], data[threadIdx.x + 128]); } __syncthreads(); }
    if (BLOCK_SIZE >= 128) { if (threadIdx.x <  64) { data[threadIdx.x] = binary_op(data[threadIdx.x], data[threadIdx.x +  64]); } __syncthreads(); }
    if (BLOCK_SIZE >=  64) { if (threadIdx.x <  32) { data[threadIdx.x] = binary_op(data[threadIdx.x], data[threadIdx.x +  32]); } __syncthreads(); }
    if (BLOCK_SIZE >=  32) { if (threadIdx.x <  16) { data[threadIdx.x] = binary_op(data[threadIdx.x], data[threadIdx.x +  16]); } __syncthreads(); }
    if (BLOCK_SIZE >=  16) { if (threadIdx.x <   8) { data[threadIdx.x] = binary_op(data[threadIdx.x], data[threadIdx.x +   8]); } __syncthreads(); }
    if (BLOCK_SIZE >=   8) { if (threadIdx.x <   4) { data[threadIdx.x] = binary_op(data[threadIdx.x], data[threadIdx.x +   4]); } __syncthreads(); }
    if (BLOCK_SIZE >=   4) { if (threadIdx.x <   2) { data[threadIdx.x] = binary_op(data[threadIdx.x], data[threadIdx.x +   2]); } __syncthreads(); }
    if (BLOCK_SIZE >=   2) { if (threadIdx.x <   1) { data[threadIdx.x] = binary_op(data[threadIdx.x], data[threadIdx.x +   1]); } __syncthreads(); }
    
    // XXX Not testing (threadIdx.x < X) for X <= 16 seems to break on G80
    // XXX This appears to be due to (illegal) instruction reorderings in the nightly builds
}

template <unsigned int BLOCK_SIZE, typename ValueIterator, typename BinaryFunction>
__device__ 
void reduce_n(ValueIterator data, const unsigned int n, BinaryFunction binary_op)
{
    if (BLOCK_SIZE >= 512) { if (threadIdx.x < 256 && threadIdx.x + 256 < n) { data[threadIdx.x] = binary_op(data[threadIdx.x], data[threadIdx.x + 256]); } __syncthreads(); }
    if (BLOCK_SIZE >= 256) { if (threadIdx.x < 128 && threadIdx.x + 128 < n) { data[threadIdx.x] = binary_op(data[threadIdx.x], data[threadIdx.x + 128]); } __syncthreads(); }
    if (BLOCK_SIZE >= 128) { if (threadIdx.x <  64 && threadIdx.x +  64 < n) { data[threadIdx.x] = binary_op(data[threadIdx.x], data[threadIdx.x +  64]); } __syncthreads(); }
    if (BLOCK_SIZE >=  64) { if (threadIdx.x <  32 && threadIdx.x +  32 < n) { data[threadIdx.x] = binary_op(data[threadIdx.x], data[threadIdx.x +  32]); } __syncthreads(); }
    if (BLOCK_SIZE >=  32) { if (threadIdx.x <  16 && threadIdx.x +  16 < n) { data[threadIdx.x] = binary_op(data[threadIdx.x], data[threadIdx.x +  16]); } __syncthreads(); }
    if (BLOCK_SIZE >=  16) { if (threadIdx.x <   8 && threadIdx.x +   8 < n) { data[threadIdx.x] = binary_op(data[threadIdx.x], data[threadIdx.x +   8]); } __syncthreads(); }
    if (BLOCK_SIZE >=   8) { if (threadIdx.x <   4 && threadIdx.x +   4 < n) { data[threadIdx.x] = binary_op(data[threadIdx.x], data[threadIdx.x +   4]); } __syncthreads(); }
    if (BLOCK_SIZE >=   4) { if (threadIdx.x <   2 && threadIdx.x +   2 < n) { data[threadIdx.x] = binary_op(data[threadIdx.x], data[threadIdx.x +   2]); } __syncthreads(); }
    if (BLOCK_SIZE >=   2) { if (threadIdx.x <   1 && threadIdx.x +   1 < n) { data[threadIdx.x] = binary_op(data[threadIdx.x], data[threadIdx.x +   1]); } __syncthreads(); }
}

} // end namespace block

} // end namespace detail

} // end namespace thrust
