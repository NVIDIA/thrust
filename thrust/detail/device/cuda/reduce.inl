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

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>
#include <thrust/reduce.h>  // for second level reduction

#include <thrust/detail/device/cuda/block/reduce.h>

namespace thrust
{

namespace detail
{

namespace device
{

namespace cuda
{

/*
 * Reduction using a single thread.  Only used for small vectors.
 *
 */
template<typename InputFunctor,
         typename OutputType,
         typename BinaryFunction>
  __global__ void
  __thrust__serial_reduce_kernel(InputFunctor input,
                                  const size_t n,
                                  OutputType * block_results,  
                                  BinaryFunction binary_op)
{
    if( threadIdx.x == 0 )
    {
        OutputType accum = input[0];
        for(unsigned int i = 1; i < n; i++){
            accum = binary_op(accum, input[i]);
        }
        block_results[0] = accum;
    }

} // end __thrust__serial_reduce_kernel()




/*
 * Reduce a vector of n elements using binary_op(op())
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
template<typename InputFunctor,
         typename OutputType,
         typename BinaryFunction,
         unsigned int BLOCK_SIZE>
  __global__ void
  __thrust__unordered_reduce_kernel(InputFunctor input,
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

    // accumulate local result
    OutputType accum = input[i];
    i += grid_size;

    while (i < n)
    {
        accum = binary_op(accum, input[i]);  
        i += grid_size;
    } 

    // copy local result to shared mem and perform reduction
    sdata[threadIdx.x] = accum;  
    
    __syncthreads(); // wait for all writes to finish
    
    if (BLOCK_SIZE >= 512) { if (threadIdx.x < 256) { sdata[threadIdx.x] = binary_op(sdata[threadIdx.x], sdata[threadIdx.x + 256]); } __syncthreads(); }
    if (BLOCK_SIZE >= 256) { if (threadIdx.x < 128) { sdata[threadIdx.x] = binary_op(sdata[threadIdx.x], sdata[threadIdx.x + 128]); } __syncthreads(); }
    if (BLOCK_SIZE >= 128) { if (threadIdx.x <  64) { sdata[threadIdx.x] = binary_op(sdata[threadIdx.x], sdata[threadIdx.x +  64]); } __syncthreads(); }
    if (BLOCK_SIZE >=  64) { if (threadIdx.x <  32) { sdata[threadIdx.x] = binary_op(sdata[threadIdx.x], sdata[threadIdx.x +  32]); } __syncthreads(); }
    if (BLOCK_SIZE >=  32) { if (threadIdx.x <  16) { sdata[threadIdx.x] = binary_op(sdata[threadIdx.x], sdata[threadIdx.x +  16]); } __syncthreads(); }

    if (BLOCK_SIZE >=  16) { if (threadIdx.x <   8) { sdata[threadIdx.x] = binary_op(sdata[threadIdx.x], sdata[threadIdx.x +   8]); } __syncthreads(); }
    if (BLOCK_SIZE >=   8) { if (threadIdx.x <   4) { sdata[threadIdx.x] = binary_op(sdata[threadIdx.x], sdata[threadIdx.x +   4]); } __syncthreads(); }
    if (BLOCK_SIZE >=   4) { if (threadIdx.x <   2) { sdata[threadIdx.x] = binary_op(sdata[threadIdx.x], sdata[threadIdx.x +   2]); } __syncthreads(); }
    if (BLOCK_SIZE >=   2) { if (threadIdx.x <   1) { sdata[threadIdx.x] = binary_op(sdata[threadIdx.x], sdata[threadIdx.x +   1]); } __syncthreads(); }

    // XXX causes the following problem on CUDA 2.2 (Komrade issue #6)
    //     Advisory: Cannot tell what pointer points to, assuming global memory space
    //accum = thrust::detail::block::reduce<OutputType,BinaryFunction,BLOCK_SIZE>
    //    (sdata, threadIdx.x, binary_op);

    // write result for this block to global mem 
    if (threadIdx.x == 0) 
        block_results[blockIdx.x] = sdata[threadIdx.x];

} // end __thrust__unordered_reduce_kernel()




template<typename InputFunctor,
         typename OutputType,
         typename BinaryFunction>
  OutputType reduce(InputFunctor input,
                    const size_t n,
                    const OutputType init,
                    BinaryFunction binary_op)
{
    const size_t BLOCK_SIZE = 256;  // BLOCK_SIZE must be a power of 2
    const size_t MAX_BLOCKS = 3 * experimental::arch::max_active_threads() / BLOCK_SIZE;

    unsigned int GRID_SIZE;

    // handle zero length array case first
    if( n == 0 )
        return init;


    // TODO if n < UINT_MAX use unsigned int instead of size_t indices in kernel

    // kernels below assume n > 0
    if( n < BLOCK_SIZE )
    {
        GRID_SIZE = 1;
    }
    else 
    {
        GRID_SIZE = std::min( (n / BLOCK_SIZE), MAX_BLOCKS);
    }

    // allocate storage for per-block results
    thrust::device_ptr<OutputType> block_results = thrust::device_malloc<OutputType>(GRID_SIZE);

    // do the gpu part
    if( n < BLOCK_SIZE )
    {
        __thrust__serial_reduce_kernel<InputFunctor, OutputType, BinaryFunction>
            <<<GRID_SIZE, 1>>>(input, n, block_results.get(), binary_op);
    } 
    else
    { 
        __thrust__unordered_reduce_kernel<InputFunctor, OutputType, BinaryFunction, BLOCK_SIZE>
            <<<GRID_SIZE, BLOCK_SIZE>>>(input, n, block_results.get(), binary_op);
    }

    // copy work array to host
    thrust::host_vector<OutputType> host_work(block_results, block_results + GRID_SIZE);

    // free device work array
    thrust::device_free(block_results);

    // reduce on the host
    return thrust::reduce(host_work.begin(), host_work.end(), init, binary_op);
} // end reduce()


} // end namespace cuda

} // end namespace device

} // end namespace detail

} // end namespace thrust

#endif // __CUDACC__

