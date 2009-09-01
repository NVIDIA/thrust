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

#include <thrust/iterator/iterator_traits.h>
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
template<unsigned int BLOCK_SIZE,
         typename InputIterator,
         typename OutputType,
         typename BinaryFunction>
  __global__ void
  __thrust__unordered_reduce_kernel(InputIterator input,
                                    const unsigned int n,
                                    OutputType * block_results,  
                                    BinaryFunction binary_op)
{
    __shared__ unsigned char sdata_workaround[BLOCK_SIZE * sizeof(OutputType)];
    OutputType *sdata = reinterpret_cast<OutputType*>(sdata_workaround);

    // perform first level of reduction,
    // write per-block results to global memory for second level reduction
    
    const unsigned int grid_size = BLOCK_SIZE * gridDim.x;
    unsigned int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    // local (per-thread) sum
    OutputType sum;
   
    // initialize local sum 
    if (i < n)
    {
        sum = input[i];
        i += grid_size;
    }

    // update sum
    while (i < n)
    {
        sum = binary_op(sum, input[i]);  
        i += grid_size;
    } 

    // copy local sum to shared memory
    sdata[threadIdx.x] = sum;  __syncthreads();

    // compute reduction across block
    thrust::detail::block::reduce_n<BLOCK_SIZE>(sdata, n, binary_op);

    // write result for this block to global mem 
    if (threadIdx.x == 0) 
        block_results[blockIdx.x] = sdata[threadIdx.x];

} // end __thrust__unordered_reduce_kernel()


// XXX if n < UINT_MAX use unsigned int instead of size_t indices in kernel


template<typename InputIterator,
         typename OutputType,
         typename BinaryFunction>
  OutputType reduce(InputIterator input,
                    const size_t n,
                    OutputType init,
                    BinaryFunction binary_op)
{
    // handle zero length array case first
    if( n == 0 )
        return init;

    // 16KB (max) - 1KB (upper bound on what's used for other purposes)
    const size_t MAX_SMEM_SIZE = 15 * 1024; 

    // largest 2^N that fits in SMEM
    static const size_t BLOCKSIZE_LIMIT1 = 1 << thrust::detail::mpl::math::log2< (MAX_SMEM_SIZE/sizeof(OutputType)) >::value;
    static const size_t BLOCKSIZE_LIMIT2 = 256;

    static const size_t BLOCK_SIZE = (BLOCKSIZE_LIMIT1 < BLOCKSIZE_LIMIT2) ? BLOCKSIZE_LIMIT1 : BLOCKSIZE_LIMIT2;
    
    const size_t MAX_BLOCKS = thrust::experimental::arch::max_active_blocks(__thrust__unordered_reduce_kernel<BLOCK_SIZE, InputIterator, OutputType, BinaryFunction>, BLOCK_SIZE, (size_t) 0);

    const unsigned int GRID_SIZE = std::max((size_t) 1, std::min( (n / BLOCK_SIZE), MAX_BLOCKS));

    // allocate storage for per-block results
    thrust::detail::raw_device_buffer<OutputType> temp(GRID_SIZE + 1);

    // set first element of temp array to init
    temp[0] = init;

    // reduce input to per-block sums
    __thrust__unordered_reduce_kernel<BLOCK_SIZE>
        <<<GRID_SIZE, BLOCK_SIZE>>>(input, n, raw_pointer_cast(&temp[1]), binary_op);

    // reduce per-block sums together with init
    __thrust__unordered_reduce_kernel<BLOCK_SIZE>
        <<<1, BLOCK_SIZE>>>(raw_pointer_cast(&temp[0]), GRID_SIZE + 1, raw_pointer_cast(&temp[0]), binary_op);

    return temp[0];
} // end reduce()


} // end namespace cuda

} // end namespace device

} // end namespace detail

} // end namespace thrust

#endif // __CUDACC__

