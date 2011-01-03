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

template<unsigned int block_size,
         typename RandomAccessIterator,
         typename BinaryFunction>
__device__
  void inplace_inclusive_scan(RandomAccessIterator first,
                              BinaryFunction binary_op)
{
  typename thrust::iterator_value<RandomAccessIterator>::type val = first[threadIdx.x];
  __syncthreads();

  if(block_size >   1) { if (threadIdx.x >=   1) { val = binary_op(first[threadIdx.x -   1], val); } __syncthreads(); first[threadIdx.x] = val; __syncthreads(); }
  if(block_size >   2) { if (threadIdx.x >=   2) { val = binary_op(first[threadIdx.x -   2], val); } __syncthreads(); first[threadIdx.x] = val; __syncthreads(); } 
  if(block_size >   4) { if (threadIdx.x >=   4) { val = binary_op(first[threadIdx.x -   4], val); } __syncthreads(); first[threadIdx.x] = val; __syncthreads(); }
  if(block_size >   8) { if (threadIdx.x >=   8) { val = binary_op(first[threadIdx.x -   8], val); } __syncthreads(); first[threadIdx.x] = val; __syncthreads(); }
  if(block_size >  16) { if (threadIdx.x >=  16) { val = binary_op(first[threadIdx.x -  16], val); } __syncthreads(); first[threadIdx.x] = val; __syncthreads(); }
  if(block_size >  32) { if (threadIdx.x >=  32) { val = binary_op(first[threadIdx.x -  32], val); } __syncthreads(); first[threadIdx.x] = val; __syncthreads(); }
  if(block_size >  64) { if (threadIdx.x >=  64) { val = binary_op(first[threadIdx.x -  64], val); } __syncthreads(); first[threadIdx.x] = val; __syncthreads(); }
  if(block_size > 128) { if (threadIdx.x >= 128) { val = binary_op(first[threadIdx.x - 128], val); } __syncthreads(); first[threadIdx.x] = val; __syncthreads(); }
  if(block_size > 256) { if (threadIdx.x >= 256) { val = binary_op(first[threadIdx.x - 256], val); } __syncthreads(); first[threadIdx.x] = val; __syncthreads(); }
  if(block_size > 512) { if (threadIdx.x >= 512) { val = binary_op(first[threadIdx.x - 512], val); } __syncthreads(); first[threadIdx.x] = val; __syncthreads(); }
} // end inplace_inclusive_scan()


template<typename RandomAccessIterator,
         typename Size,
         typename BinaryFunction>
__device__
void inplace_inclusive_scan_n(RandomAccessIterator first,
                              Size n,
                              BinaryFunction binary_op)
{
  typename thrust::iterator_value<RandomAccessIterator>::type val = first[threadIdx.x];
  __syncthreads();

  // assume n <= 2048
  if(n >    1) { if (threadIdx.x < n && threadIdx.x >=    1) { val = binary_op(first[threadIdx.x -    1], val); } __syncthreads(); first[threadIdx.x] = val; __syncthreads(); }
  if(n >    2) { if (threadIdx.x < n && threadIdx.x >=    2) { val = binary_op(first[threadIdx.x -    2], val); } __syncthreads(); first[threadIdx.x] = val; __syncthreads(); } 
  if(n >    4) { if (threadIdx.x < n && threadIdx.x >=    4) { val = binary_op(first[threadIdx.x -    4], val); } __syncthreads(); first[threadIdx.x] = val; __syncthreads(); }
  if(n >    8) { if (threadIdx.x < n && threadIdx.x >=    8) { val = binary_op(first[threadIdx.x -    8], val); } __syncthreads(); first[threadIdx.x] = val; __syncthreads(); }
  if(n >   16) { if (threadIdx.x < n && threadIdx.x >=   16) { val = binary_op(first[threadIdx.x -   16], val); } __syncthreads(); first[threadIdx.x] = val; __syncthreads(); }
  if(n >   32) { if (threadIdx.x < n && threadIdx.x >=   32) { val = binary_op(first[threadIdx.x -   32], val); } __syncthreads(); first[threadIdx.x] = val; __syncthreads(); }
  if(n >   64) { if (threadIdx.x < n && threadIdx.x >=   64) { val = binary_op(first[threadIdx.x -   64], val); } __syncthreads(); first[threadIdx.x] = val; __syncthreads(); }
  if(n >  128) { if (threadIdx.x < n && threadIdx.x >=  128) { val = binary_op(first[threadIdx.x -  128], val); } __syncthreads(); first[threadIdx.x] = val; __syncthreads(); }
  if(n >  256) { if (threadIdx.x < n && threadIdx.x >=  256) { val = binary_op(first[threadIdx.x -  256], val); } __syncthreads(); first[threadIdx.x] = val; __syncthreads(); }
  if(n >  512) { if (threadIdx.x < n && threadIdx.x >=  512) { val = binary_op(first[threadIdx.x -  512], val); } __syncthreads(); first[threadIdx.x] = val; __syncthreads(); }
  if(n > 1024) { if (threadIdx.x < n && threadIdx.x >= 1024) { val = binary_op(first[threadIdx.x - 1024], val); } __syncthreads(); first[threadIdx.x] = val; __syncthreads(); }
} // end inplace_inclusive_scan()

} // end namespace block
} // end namespace cuda
} // end namespace device
} // end namespace detail
} // end namespace thrust

