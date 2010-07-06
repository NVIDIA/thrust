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




#if THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_MSVC
// temporarily disable 'possible loss of data' warnings on MSVC
#pragma warning(push)
#pragma warning(disable : 4244 4267)
#endif

template<typename RandomAccessIterator,
         typename SizeType,
         typename OutputType,
         typename BinaryFunction>
  thrust::pair<SizeType,SizeType>
    get_unordered_blocked_reduce_n_schedule(RandomAccessIterator first,
                                            SizeType n,
                                            OutputType init,
                                            BinaryFunction binary_op)
{
  // decide whether or not we will use smem
  size_t smem_per_thread = 0;
  if(sizeof(OutputType) <= 64)
  {
    smem_per_thread = sizeof(OutputType);
  }

  // choose block_size
  size_t block_size = 0;
  if(smem_per_thread > 0)
  {
    block_size = thrust::experimental::arch::max_blocksize_with_highest_occupancy(reduce_n_smem<RandomAccessIterator, OutputType, BinaryFunction>, smem_per_thread);
  }
  else
  {
    block_size = thrust::experimental::arch::max_blocksize_with_highest_occupancy(reduce_n_gmem<RandomAccessIterator, OutputType, BinaryFunction>, smem_per_thread);
  }

  const size_t smem_size = block_size * smem_per_thread;

  // choose the maximum number of blocks we can launch
  size_t max_blocks = 0;
  if(smem_per_thread > 0)
  {
    max_blocks = thrust::experimental::arch::max_active_blocks(reduce_n_smem<RandomAccessIterator, OutputType, BinaryFunction>, block_size, smem_size);
  }
  else
  {
    max_blocks = thrust::experimental::arch::max_active_blocks(reduce_n_gmem<RandomAccessIterator, OutputType, BinaryFunction>, block_size, smem_size);
  }

  // finalize the number of blocks to launch
  const size_t num_blocks = std::min<size_t>(max_blocks, (n + (block_size - 1)) / block_size);

  return thrust::make_pair(num_blocks,block_size);
}


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
    const thrust::pair<size_t,size_t> blocking = thrust::detail::device::cuda::detail::get_unordered_blocked_reduce_n_schedule(first,n,init,binary_op);
    const size_t num_blocks = blocking.first;
    const size_t block_size = blocking.second;

    // XXX avoid recomputing this here -- could be returned by the above call
    const size_t smem_per_thread = sizeof(OutputType);
    const size_t smem_size = block_size * smem_per_thread;

    // allocate storage for per-block results
    thrust::detail::raw_cuda_device_buffer<OutputType> temp(num_blocks + 1);

    // set first element of temp array to init
    temp[0] = init;

    // reduce input to per-block sums
    reduce_n_smem<<<num_blocks, block_size, smem_size>>>(first, n, raw_pointer_cast(&temp[1]), binary_op);

    // reduce per-block sums together with init
    {
#if CUDA_VERSION >= 3000
        const size_t block_size_pass2 = thrust::experimental::arch::max_blocksize(reduce_n_smem<OutputType *, OutputType, BinaryFunction>, smem_per_thread);
#else
        const size_t block_size_pass2 = 32;
#endif        
        const size_t smem_size_pass2  = smem_per_thread * block_size_pass2;
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
    const thrust::pair<size_t,size_t> blocking = thrust::detail::device::cuda::detail::get_unordered_blocked_reduce_n_schedule(first,n,init,binary_op);
    const size_t num_blocks = blocking.first;
    const size_t block_size = blocking.second;

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
        const size_t block_size_pass2 = std::min(block_size, thrust::experimental::arch::max_blocksize(reduce_n_gmem<OutputType *, OutputType, BinaryFunction>, 0));
#else
        const size_t block_size_pass2 = 32;
#endif        
        detail::reduce_n_gmem<<<1, block_size_pass2>>>(raw_pointer_cast(&temp[0]), num_blocks + 1, raw_pointer_cast(&temp[0]), raw_pointer_cast(&shared_array[0]), binary_op);
    }
    
    return temp[0];
} // end reduce_n()


template<typename RandomAccessIterator1,
         typename SizeType,
         typename BlockingPair,
         typename BinaryFunction,
         typename RandomAccessIterator2>
  void unordered_blocked_reduce_n(RandomAccessIterator1 first,
                                  SizeType n,
                                  BlockingPair blocking,
                                  BinaryFunction binary_op,
                                  RandomAccessIterator2 result,
                                  thrust::detail::true_type)   // reduce in shared memory
{
  typedef typename thrust::iterator_value<RandomAccessIterator2>::type OutputType;

  const size_t num_blocks = blocking.first;
  const size_t block_size = blocking.second;

  const size_t smem_per_thread = sizeof(OutputType);
  const size_t smem_size = block_size * smem_per_thread;

  // reduce input to per-block sums
  reduce_n_smem<<<num_blocks, block_size, smem_size>>>(first, n, thrust::raw_pointer_cast(&*result), binary_op);
} // end unordered_blocked_reduce_n()


template<typename RandomAccessIterator1,
         typename SizeType,
         typename BlockingPair,
         typename BinaryFunction,
         typename RandomAccessIterator2>
  void unordered_blocked_reduce_n(RandomAccessIterator1 first,
                                  SizeType n,
                                  BlockingPair blocking,
                                  BinaryFunction binary_op,
                                  RandomAccessIterator2 result,
                                  thrust::detail::false_type)   // reduce in global memory
{
  typedef typename thrust::iterator_value<RandomAccessIterator2>::type OutputType;

  const size_t num_blocks = blocking.first;
  const size_t block_size = blocking.second;

  // allocate storage for shared array
  thrust::detail::raw_cuda_device_buffer<OutputType> shared_array(block_size * num_blocks);

  // reduce input to per-block sums
  detail::reduce_n_gmem<<<num_blocks, block_size>>>(first, n, raw_pointer_cast(&*result), raw_pointer_cast(&shared_array[0]), binary_op);
} // end unordered_blocked_reduce_n()


} // end namespace detail



#if THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_MSVC
// reenable 'possible loss of data' warnings
#pragma warning(pop)
#endif



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

template<typename RandomAccessIterator1,
         typename SizeType,
         typename BlockingPair,
         typename BinaryFunction,
         typename RandomAccessIterator2>
  void unordered_blocked_reduce_n(RandomAccessIterator1 first,
                                  SizeType n,
                                  BlockingPair blocking,
                                  BinaryFunction binary_op,
                                  RandomAccessIterator2 result)
{
  typedef typename thrust::iterator_value<RandomAccessIterator2>::type OutputType;

  // handle zero length input or output
  if(n == 0 || blocking.first == 0)
    return;

  // whether to perform blockwise reductions in shared memory or global memory
  thrust::detail::integral_constant<bool, sizeof(OutputType) <= 64> use_smem;

  return detail::unordered_blocked_reduce_n(first, n, blocking, binary_op, result, use_smem);
} // end unordered_blocked_reduce_n()

} // end namespace cuda
} // end namespace device
} // end namespace detail
} // end namespace thrust

#endif // THRUST_DEVICE_COMPILER != THRUST_DEVICE_COMPILER_NVCC

