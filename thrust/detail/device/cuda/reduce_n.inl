/*
 *  Copyright 2008-2010 NVIDIA Corporation
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

// do not attempt to compile this file with any other compiler
#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC

// to configure launch parameters
#include <thrust/experimental/arch.h>

#include <thrust/detail/type_traits.h>

#include <thrust/detail/raw_buffer.h>
#include <thrust/detail/device/cuda/block/reduce.h>
#include <thrust/detail/device/cuda/extern_shared_ptr.h>

namespace thrust
{
namespace detail
{
namespace device
{
namespace cuda
{
namespace detail
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

template<typename InputIterator,
         typename OutputType,
         typename BinaryFunction,
         typename SharedArray>
  __device__ void
  reduce_n_device(InputIterator input,
                  const unsigned int n,
                  OutputType * block_results,  
                  BinaryFunction binary_op,
                  SharedArray shared_array)
{
    // perform one level of reduction, writing per-block results to 
    // global memory for subsequent processing (e.g. second level reduction) 
    const unsigned int grid_size = blockDim.x * gridDim.x;
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

    // advance input
    input += i;

    if (i < n)
    {
        // initialize local sum 
        OutputType sum = thrust::detail::device::dereference(input);

        i     += grid_size;
        input += grid_size;

        // accumulate local sum
        while (i < n)
        {
            OutputType val = thrust::detail::device::dereference(input);
            sum = binary_op(sum, val);

            i     += grid_size;
            input += grid_size;
        }

        // copy local sum to shared memory
        shared_array[threadIdx.x] = sum;
    }

    __syncthreads();

    // compute reduction across block
    thrust::detail::device::cuda::block::reduce_n(shared_array, min(n - blockDim.x * blockIdx.x, blockDim.x), binary_op);

    // write result for this block to global mem 
    if (threadIdx.x == 0) 
        block_results[blockIdx.x] = shared_array[threadIdx.x];

} // end reduce_n_device()

template<typename InputIterator,
         typename OutputType,
         typename BinaryFunction>
  __global__ void
  reduce_n_smem(InputIterator input,
                const unsigned int n,
                OutputType * block_results,  
                BinaryFunction binary_op)
{
    thrust::detail::device::cuda::extern_shared_ptr<OutputType> shared_ptr;
    OutputType *shared_array = shared_ptr;

    reduce_n_device(input, n, block_results, binary_op, shared_array);
} // end reduce_n_kernel()


template<typename InputIterator,
         typename OutputType,
         typename BinaryFunction>
  __global__ void
  reduce_n_gmem(InputIterator input,
                const unsigned int n,
                OutputType * block_results,  
                OutputType * shared_array,
                BinaryFunction binary_op)
{
    reduce_n_device(input, n, block_results, binary_op, shared_array + blockDim.x * blockIdx.x);
} // end reduce_n_kernel()

template<typename InputIterator,
         typename SizeType,
         typename OutputType,
         typename BinaryFunction>
  OutputType reduce_n(InputIterator first,
                      SizeType n,
                      OutputType init,
                      BinaryFunction binary_op,
                      thrust::detail::true_type)    // reduce in shared memory
{
    // determine launch parameters
    const size_t smem_per_thread = sizeof(OutputType);
    const size_t block_size = thrust::experimental::arch::max_blocksize_with_highest_occupancy(reduce_n_smem<InputIterator, OutputType, BinaryFunction>, smem_per_thread);
    const size_t smem_size  = block_size * smem_per_thread;
    const size_t max_blocks = thrust::experimental::arch::max_active_blocks(reduce_n_smem<InputIterator, OutputType, BinaryFunction>, block_size, smem_size);
    const size_t num_blocks = std::min(max_blocks, (n + (block_size - 1)) / block_size);

    // allocate storage for per-block results
    thrust::detail::raw_cuda_device_buffer<OutputType> temp(num_blocks + 1);

    // set first element of temp array to init
    temp[0] = init;

    // reduce input to per-block sums
    reduce_n_smem<<<num_blocks, block_size, smem_size>>>(first, n, raw_pointer_cast(&temp[1]), binary_op);

    // reduce per-block sums together with init
    {
#if CUDA_VERSION >= 3000
        const unsigned int block_size_pass2 = thrust::experimental::arch::max_blocksize(reduce_n_smem<OutputType *, OutputType, BinaryFunction>, smem_per_thread);
#else
        const unsigned int block_size_pass2 = 32;
#endif        
        const unsigned int smem_size_pass2  = smem_per_thread * block_size_pass2;
        reduce_n_smem<<<1, block_size_pass2, smem_size_pass2>>>(raw_pointer_cast(&temp[0]), num_blocks + 1, raw_pointer_cast(&temp[0]), binary_op);
    }

    return temp[0];
} // end reduce_n()


template<typename InputIterator,
         typename SizeType,
         typename OutputType,
         typename BinaryFunction>
  OutputType reduce_n(InputIterator first,
                      SizeType n,
                      OutputType init,
                      BinaryFunction binary_op,
                      thrust::detail::false_type)    // reduce in global memory
{
    // determine launch parameters
    const size_t smem_per_thread = 0;
    const size_t block_size = thrust::experimental::arch::max_blocksize_with_highest_occupancy(reduce_n_gmem<InputIterator, OutputType, BinaryFunction>, smem_per_thread);
    const size_t smem_size  = block_size * smem_per_thread;
    const size_t max_blocks = thrust::experimental::arch::max_active_blocks(reduce_n_gmem<InputIterator, OutputType, BinaryFunction>, block_size, smem_size);
    const size_t num_blocks = std::min(max_blocks, (n + (block_size - 1)) / block_size);

    // allocate storage for per-block results
    thrust::detail::raw_cuda_device_buffer<OutputType> temp(num_blocks + 1);

    // allocate storage for shared array
    thrust::detail::raw_cuda_device_buffer<OutputType> shared_array(block_size * num_blocks);

    // set first element of temp array to init
    temp[0] = init;

    // reduce input to per-block sums
    detail::reduce_n_gmem<<<num_blocks, block_size, 0>>>(first, n, raw_pointer_cast(&temp[1]), raw_pointer_cast(&shared_array[0]), binary_op);
    
    // reduce per-block sums together with init
    {
#if CUDA_VERSION >= 3000
        const unsigned int block_size_pass2 = std::min(block_size, thrust::experimental::arch::max_blocksize(reduce_n_gmem<OutputType *, OutputType, BinaryFunction>, smem_per_thread));
#else
        const unsigned int block_size_pass2 = 32;
#endif        
        const unsigned int smem_size_pass2  = smem_per_thread * block_size_pass2;
        detail::reduce_n_gmem<<<1, block_size_pass2, smem_size_pass2>>>(raw_pointer_cast(&temp[0]), num_blocks + 1, raw_pointer_cast(&temp[0]), raw_pointer_cast(&shared_array[0]), binary_op);
    }
    
    return temp[0];
} // end reduce_n()

} // end namespace detail


// TODO add runtime switch for SizeType vs. unsigned int
// TODO use closure approach to handle large iterators & functors (i.e. sum > 256 bytes)

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

    // whether to perform blockwise reductions in shared memory or global memory
    thrust::detail::integral_constant<bool, sizeof(OutputType) <= 64> use_smem;

    return detail::reduce_n(first, n, init, binary_op, use_smem);
} // end reduce_n()

} // end namespace cuda
} // end namespace device
} // end namespace detail
} // end namespace thrust

#endif // THRUST_DEVICE_COMPILER != THRUST_DEVICE_COMPILER_NVCC

