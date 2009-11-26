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


/*! \file reduce.inl
 *  \brief Inline file for reduce.h
 */

#pragma once

// do not attempt to compile this file with any other compiler
#ifdef __CUDACC__

#include <thrust/detail/raw_buffer.h>

#include <thrust/experimental/arch.h>

#include <thrust/detail/device/cuda/block/reduce.h>

#include <thrust/detail/mpl/math.h> // for log2<N>

namespace thrust
{
namespace detail
{
namespace device
{
namespace cuda
{

/*
 * Reduce a vector of n elements using binary_op()
 *
 * The order of reduction is not defined, so binary_op() should
 * be a commutative (and associative) operator such as 
 * (integer) addition.  Since floating point operations
 * do not completely satisfy these criteria, the result is 
 * generally not the same as a consecutive reduction of 
 * the elements.
 * 
 * Uses the same pattern as reduce6() in the CUDA SDK
 *
 */

extern __shared__ char sdata_workaround[];
template<typename InputIterator,
         typename OutputType,
         typename BinaryFunction>
  __global__ void
  reduce_n_kernel(InputIterator input,
                  const unsigned int n,
                  OutputType * block_results,  
                  BinaryFunction binary_op)
{
    OutputType *sdata = reinterpret_cast<OutputType*>(sdata_workaround);

    // perform first level of reduction,
    // write per-block results to global memory for second level reduction
    
    const unsigned int grid_size = blockDim.x * gridDim.x;
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

    // local (per-thread) sum
    OutputType sum;
   
    // initialize local sum 
    if (i < n)
    {
        sum = thrust::detail::device::dereference(input, i);
        i += grid_size;
    }

    // update sum
    while (i < n)
    {
        sum = binary_op(sum, thrust::detail::device::dereference(input, i));
        i += grid_size;
    } 

    // copy local sum to shared memory
    sdata[threadIdx.x] = sum;  __syncthreads();

    // compute reduction across block
    block::reduce_n(sdata, min(n, blockDim.x), binary_op);

    // write result for this block to global mem 
    if (threadIdx.x == 0) 
        block_results[blockIdx.x] = sdata[threadIdx.x];

} // end reduce_n_kernel()


template<typename InputIterator,
         typename SizeType,
         typename OutputType,
         typename BinaryFunction>
  OutputType reduce_n(InputIterator first,
                      SizeType n,
                      OutputType init,
                      BinaryFunction binary_op)
{
    // handle zero length array case first
    if( n == 0 )
        return init;
   
    // determine launch parameters
    const size_t block_size = thrust::experimental::arch::max_blocksize_with_highest_occupancy(reduce_n_kernel<InputIterator, OutputType, BinaryFunction>, sizeof(OutputType));
    const size_t max_blocks = thrust::experimental::arch::max_active_blocks(reduce_n_kernel<InputIterator, OutputType, BinaryFunction>, block_size, 0);
    const size_t num_blocks = std::min(max_blocks, std::max((size_t) 1, n / block_size));
    const size_t smem_size  = block_size * sizeof(OutputType);

    // allocate storage for per-block results
    thrust::detail::raw_device_buffer<OutputType> temp(num_blocks + 1);

    // set first element of temp array to init
    temp[0] = init;

    // reduce input to per-block sums
    reduce_n_kernel<<<num_blocks, block_size, smem_size>>>(first, n, raw_pointer_cast(&temp[1]), binary_op);

    // reduce per-block sums together with init
    reduce_n_kernel<<<1, block_size, smem_size>>>(raw_pointer_cast(&temp[0]), num_blocks + 1, raw_pointer_cast(&temp[0]), binary_op);

    return temp[0];
} // end reduce_n()

} // end namespace cuda
} // end namespace device
} // end namespace detail
} // end namespace thrust

#endif // __CUDACC__

