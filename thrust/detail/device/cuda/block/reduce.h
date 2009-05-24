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

template <typename ValueType, typename BinaryFunction, unsigned int BLOCK_SIZE>
__device__ ValueType
reduce(ValueType * data, const unsigned int tid, BinaryFunction binary_op)
{
    if (BLOCK_SIZE >= 512) { if (tid < 256) { data[tid] = binary_op(data[tid], data[tid + 256]); } __syncthreads(); }
    if (BLOCK_SIZE >= 256) { if (tid < 128) { data[tid] = binary_op(data[tid], data[tid + 128]); } __syncthreads(); }
    if (BLOCK_SIZE >= 128) { if (tid <  64) { data[tid] = binary_op(data[tid], data[tid +  64]); } __syncthreads(); }
    if (BLOCK_SIZE >=  64) { if (tid <  32) { data[tid] = binary_op(data[tid], data[tid +  32]); } __syncthreads(); }
    if (BLOCK_SIZE >=  32) { if (tid <  16) { data[tid] = binary_op(data[tid], data[tid +  16]); } __syncthreads(); }

    if (BLOCK_SIZE >=  16) { if (tid <   8) { data[tid] = binary_op(data[tid], data[tid +   8]); } __syncthreads(); }
    if (BLOCK_SIZE >=   8) { if (tid <   4) { data[tid] = binary_op(data[tid], data[tid +   4]); } __syncthreads(); }
    if (BLOCK_SIZE >=   4) { if (tid <   2) { data[tid] = binary_op(data[tid], data[tid +   2]); } __syncthreads(); }
    if (BLOCK_SIZE >=   2) { if (tid <   1) { data[tid] = binary_op(data[tid], data[tid +   1]); } __syncthreads(); }
    
    // XXX Not synchronizing here seems to break this kernel in general
    // XXX Not testing (tid < X) for X <= 16 seems to break on G80
    // XXX This appears to be due to (illegal) instruction reorderings in the nightly builds

    return data[0];
}

} // end namespace block

} // end namespace detail

} // end namespace thrust
