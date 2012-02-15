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


/*! \file reduce.inl
 *  \brief Inline file for reduce.h
 */

#include <thrust/iterator/iterator_traits.h>

#include <thrust/detail/config.h>
#include <thrust/detail/minmax.h>
#include <thrust/detail/uninitialized_array.h>

#include <thrust/detail/backend/dereference.h>
#include <thrust/detail/backend/cuda/arch.h>
#include <thrust/detail/backend/cuda/extern_shared_ptr.h>
#include <thrust/detail/backend/cuda/block/reduce.h>
#include <thrust/detail/backend/cuda/detail/launch_closure.h>
#include <thrust/detail/backend/cuda/detail/launch_calculator.h>

namespace thrust
{
namespace detail
{
namespace backend
{
namespace cuda
{

// guard built-in CUDA variables and types
#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC

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
template <typename InputIterator,
          typename Size,
          typename T,
          typename OutputIterator,
          typename BinaryFunction>
struct unordered_reduce_closure
{
  InputIterator  input;
  Size           n;
  T              init;
  OutputIterator output;
  BinaryFunction binary_op;
  unsigned int shared_array_size;

  unordered_reduce_closure(InputIterator input, Size n, T init, OutputIterator output, BinaryFunction binary_op, unsigned int shared_array_size)
    : input(input), n(n), init(init), output(output), binary_op(binary_op), shared_array_size(shared_array_size) {}

  __device__ 
  void operator()(void)
  {
    typedef typename thrust::iterator_value<OutputIterator>::type OutputType;
    thrust::detail::backend::cuda::extern_shared_ptr<OutputType>  shared_array;

    Size grid_size = blockDim.x * gridDim.x;

    Size i = blockDim.x * blockIdx.x + threadIdx.x;
      
    input += i;

    // compute reduction with all blockDim.x threads
    OutputType sum = dereference(input);

    i     += grid_size;
    input += grid_size;

    while (i < n)
    {
      OutputType val = dereference(input);

      sum = binary_op(sum, val);

      i      += grid_size;
      input  += grid_size;
    }

    // write first shared_array_size values into shared memory
    if (threadIdx.x < shared_array_size)
      shared_array[threadIdx.x] = sum;  

    // accumulate remaining values (if any) to shared memory in stages
    if (blockDim.x > shared_array_size)
    {
      unsigned int lb = shared_array_size;
      unsigned int ub = shared_array_size + lb;
      
      while (lb < blockDim.x)
      {
        __syncthreads();

        if (lb <= threadIdx.x && threadIdx.x < ub)
        {
          OutputType tmp = shared_array[threadIdx.x - lb];
          shared_array[threadIdx.x - lb] = binary_op(tmp, sum);
        }

        lb += shared_array_size;
        ub += shared_array_size;
      }
    }
    
    __syncthreads(); 

    thrust::detail::backend::cuda::block::reduce_n(shared_array, thrust::min<unsigned int>(blockDim.x, shared_array_size), binary_op);
  
    if (threadIdx.x == 0)
    {
      OutputType tmp = shared_array[0];

      if (gridDim.x == 1)
        tmp = binary_op(init, tmp);

      output += blockIdx.x;
      dereference(output) = tmp;
    }
  }
};


__THRUST_DISABLE_MSVC_POSSIBLE_LOSS_OF_DATA_WARNING_BEGIN

template<typename InputIterator,
         typename Size,
         typename OutputType,
         typename BinaryFunction>
  OutputType reduce_n(InputIterator first,
                      Size n,
                      OutputType init,
                      BinaryFunction binary_op)
{
  // we're attempting to launch a kernel, assert we're compiling with nvcc
  // ========================================================================
  // X Note to the user: If you've found this line due to a compiler error, X
  // X you need to compile your code using nvcc, rather than g++ or cl.exe  X
  // ========================================================================
  THRUST_STATIC_ASSERT( (depend_on_instantiation<InputIterator, THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC>::value) );

  if (n == 0)
    return init;

  typedef          uninitialized_array<OutputType, thrust::detail::cuda_device_space_tag> OutputArray;
  typedef typename OutputArray::iterator OutputIterator;

  typedef unordered_reduce_closure<InputIterator,Size,OutputType,OutputIterator,BinaryFunction> Closure;
    
  arch::function_attributes_t attributes = thrust::detail::backend::cuda::detail::closure_attributes<Closure>();
  
  // TODO chose this in a more principled manner
  size_t threshold = thrust::max<size_t>(2 * attributes.maxThreadsPerBlock, 1024);

  arch::device_properties_t properties = thrust::detail::backend::cuda::arch::device_properties();

  // launch configuration
  size_t num_blocks; 
  size_t block_size; 
  size_t array_size; 
  size_t smem_bytes; 

  // first level reduction
  if (static_cast<size_t>(n) < threshold)
  {
    num_blocks = 1;
    block_size = thrust::min(static_cast<size_t>(n), static_cast<size_t>(attributes.maxThreadsPerBlock));
    array_size = thrust::min(block_size, (properties.sharedMemPerBlock - attributes.sharedSizeBytes) / sizeof(OutputType));
    smem_bytes = sizeof(OutputType) * array_size;
  }
  else
  {
    thrust::detail::backend::cuda::detail::launch_calculator<Closure> calculator;
    
    thrust::tuple<size_t,size_t,size_t> config = calculator.with_variable_block_size_available_smem();

    num_blocks = thrust::min(thrust::get<0>(config), static_cast<size_t>(n) / thrust::get<1>(config));
    block_size = thrust::get<1>(config);
    array_size = thrust::min(block_size, thrust::get<2>(config) / sizeof(OutputType));
    smem_bytes = sizeof(OutputType) * array_size;
  }
 
  // TODO assert(n <= num_blocks * block_size);
  // TODO if (shared_array_size < 1) throw cuda exception "insufficient shared memory"

  OutputArray output(num_blocks);

  Closure closure(first, n, init, output.begin(), binary_op, array_size);
  
  //std::cout << "Launching " << num_blocks << " blocks of kernel with " << block_size << " threads and " << smem_bytes << " shared memory per block " << std::endl;

  thrust::detail::backend::cuda::detail::launch_closure(closure, num_blocks, block_size, smem_bytes);

  // second level reduction
  if (num_blocks > 1)
  {
    typedef unordered_reduce_closure<OutputIterator,Size,OutputType,OutputIterator,BinaryFunction> Closure;

    arch::function_attributes_t attributes = thrust::detail::backend::cuda::detail::closure_attributes<Closure>();

    num_blocks = 1;
    block_size = thrust::min(output.size(), static_cast<size_t>(attributes.maxThreadsPerBlock));
    array_size = thrust::min(block_size, (properties.sharedMemPerBlock - attributes.sharedSizeBytes) / sizeof(OutputType));
    smem_bytes = sizeof(OutputType) * array_size;
  
    // TODO if (shared_array_size < 1) throw cuda exception "insufficient shared memory"

    Closure closure(output.begin(), output.size(), init, output.begin(), binary_op, array_size);

    //std::cout << "Launching " << num_blocks << " blocks of kernel with " << block_size << " threads and " << smem_bytes << " shared memory per block " << std::endl;

    thrust::detail::backend::cuda::detail::launch_closure(closure, num_blocks, block_size, smem_bytes);
  }
  
  return output[0];
}

__THRUST_DISABLE_MSVC_POSSIBLE_LOSS_OF_DATA_WARNING_END

#endif // THRUST_DEVICE_COMPILER_NVCC

} // end namespace cuda
} // end namespace backend
} // end namespace detail
} // end namespace thrust

