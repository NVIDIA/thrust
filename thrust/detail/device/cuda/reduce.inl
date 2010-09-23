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
#include <thrust/detail/device/cuda/arch.h>

#include <thrust/detail/type_traits.h>

#include <thrust/detail/raw_buffer.h>
#include <thrust/detail/device/cuda/block/reduce.h>
#include <thrust/detail/device/cuda/extern_shared_ptr.h>
#include <thrust/detail/device/cuda/dispatch/reduce.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/functional.h>
#include <thrust/detail/device/cuda/synchronize.h>

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


template <typename InputType, typename OutputType, typename BinaryFunction, typename WideType>
  struct wide_unary_op : public thrust::unary_function<WideType,OutputType>
{
  BinaryFunction binary_op;

  __host__ __device__ 
  wide_unary_op(BinaryFunction binary_op) 
    : binary_op(binary_op) {}

  __host__ __device__
  OutputType operator()(WideType x)
  {
    WideType mask = ((WideType) 1 << (8 * sizeof(InputType))) - 1;

    OutputType sum = static_cast<InputType>(x & mask);

    for(unsigned int n = 1; n < sizeof(WideType) / sizeof(InputType); n++)
      sum = binary_op(sum, static_cast<InputType>( (x >> (8 * n * sizeof(InputType))) & mask ) );

    return sum;
  }
};


#if THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_MSVC
// temporarily disable 'possible loss of data' warnings on MSVC
#pragma warning(push)
#pragma warning(disable : 4244 4267)
#endif


template<typename RandomAccessIterator1,
         typename SizeType1,
         typename SizeType2,
         typename BinaryFunction,
         typename RandomAccessIterator2>
  void unordered_blocked_reduce_n(RandomAccessIterator1 first,
                                  SizeType1 n,
                                  SizeType2 num_blocks,
                                  BinaryFunction binary_op,
                                  RandomAccessIterator2 result,
                                  thrust::detail::true_type)   // reduce in shared memory
{
  typedef typename thrust::iterator_value<RandomAccessIterator2>::type OutputType;

  // determine launch parameters
  const size_t smem_per_thread = sizeof(OutputType);
  const size_t block_size = thrust::detail::device::cuda::arch::max_blocksize_with_highest_occupancy(reduce_n_smem<RandomAccessIterator1, OutputType, BinaryFunction>, smem_per_thread);
  const size_t smem_size = block_size * smem_per_thread;

  // reduce input to per-block sums
  reduce_n_smem<<<num_blocks, block_size, smem_size>>>(first, n, thrust::raw_pointer_cast(&*result), binary_op);
  synchronize_if_enabled("reduce_n_smem");
} // end unordered_blocked_reduce_n()


template<typename RandomAccessIterator1,
         typename SizeType1,
         typename SizeType2,
         typename BinaryFunction,
         typename RandomAccessIterator2>
  void unordered_blocked_reduce_n(RandomAccessIterator1 first,
                                  SizeType1 n,
                                  SizeType2 num_blocks,
                                  BinaryFunction binary_op,
                                  RandomAccessIterator2 result,
                                  thrust::detail::false_type)   // reduce in global memory
{
  typedef typename thrust::iterator_value<RandomAccessIterator2>::type OutputType;

  const size_t block_size = thrust::detail::device::cuda::arch::max_blocksize_with_highest_occupancy(reduce_n_gmem<RandomAccessIterator1, OutputType, BinaryFunction>, 0);

  // allocate storage for shared array
  thrust::detail::raw_cuda_device_buffer<OutputType> shared_array(block_size * num_blocks);

  // reduce input to per-block sums
  detail::reduce_n_gmem<<<num_blocks, block_size>>>(first, n, raw_pointer_cast(&*result), raw_pointer_cast(&shared_array[0]), binary_op);
  synchronize_if_enabled("reduce_n_gmem");
} // end unordered_blocked_reduce_n()


template<typename Iterator, typename InputType = typename thrust::iterator_value<Iterator>::type>
  struct use_wide_reduction
    : thrust::detail::integral_constant<
        bool,
        thrust::detail::is_pod<InputType>::value
        && thrust::detail::is_trivial_iterator<Iterator>::value
        && (sizeof(InputType) == 1 || sizeof(InputType) == 2)
      >
{};


} // end namespace detail



#if THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_MSVC
// reenable 'possible loss of data' warnings
#pragma warning(pop)
#endif


template<typename RandomAccessIterator,
         typename SizeType,
         typename OutputType,
         typename BinaryFunction>
  SizeType get_unordered_blocked_wide_reduce_n_schedule(RandomAccessIterator first,
                                                        SizeType n,
                                                        OutputType init,
                                                        BinaryFunction binary_op)
{
  // "wide" reduction for small types like char, short, etc.
  typedef typename thrust::iterator_traits<RandomAccessIterator>::value_type InputType;
  typedef unsigned int WideType;

  const size_t input_type_per_wide_type = sizeof(WideType) / sizeof(InputType);
  const size_t n_wide = n / input_type_per_wide_type;

  thrust::device_ptr<const WideType> wide_first = thrust::device_pointer_cast(reinterpret_cast<const WideType *>(thrust::raw_pointer_cast(&*first)));

  thrust::transform_iterator<
    detail::wide_unary_op<InputType,OutputType,BinaryFunction,WideType>,
    thrust::device_ptr<const WideType>
  > xfrm_wide_first = thrust::make_transform_iterator(wide_first, detail::wide_unary_op<InputType,OutputType,BinaryFunction,WideType>(binary_op));

  const size_t num_blocks_from_wide_part =
    thrust::detail::device::cuda::get_unordered_blocked_standard_reduce_n_schedule(xfrm_wide_first, n_wide, init, binary_op);

  // add one block to reduce the tail (if there is one)
  RandomAccessIterator tail_first = first + n_wide * input_type_per_wide_type;
  const size_t n_tail = n - (tail_first - first);

  return num_blocks_from_wide_part + ((n_tail > 0) ? 1 : 0);
}


template<typename RandomAccessIterator,
         typename SizeType,
         typename OutputType,
         typename BinaryFunction>
  SizeType get_unordered_blocked_standard_reduce_n_schedule(RandomAccessIterator first,
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
    block_size = thrust::detail::device::cuda::arch::max_blocksize_with_highest_occupancy(detail::reduce_n_smem<RandomAccessIterator, OutputType, BinaryFunction>, smem_per_thread);
  }
  else
  {
    block_size = thrust::detail::device::cuda::arch::max_blocksize_with_highest_occupancy(detail::reduce_n_gmem<RandomAccessIterator, OutputType, BinaryFunction>, smem_per_thread);
  }

  const size_t smem_size = block_size * smem_per_thread;

  // choose the maximum number of blocks we can launch
  size_t max_blocks = 0;
  if(smem_per_thread > 0)
  {
    max_blocks = thrust::detail::device::cuda::arch::max_active_blocks(detail::reduce_n_smem<RandomAccessIterator, OutputType, BinaryFunction>, block_size, smem_size);
  }
  else
  {
    max_blocks = thrust::detail::device::cuda::arch::max_active_blocks(detail::reduce_n_gmem<RandomAccessIterator, OutputType, BinaryFunction>, block_size, smem_size);
  }

  // finalize the number of blocks to launch
  const size_t num_blocks = std::min<size_t>(max_blocks, (n + (block_size - 1)) / block_size);

  return num_blocks;
}


// TODO add runtime switch for SizeType vs. unsigned int
// TODO use closure approach to handle large iterators & functors (i.e. sum > 256 bytes)

template<typename RandomAccessIterator1,
         typename SizeType1,
         typename SizeType2,
         typename BinaryFunction,
         typename RandomAccessIterator2>
  void unordered_blocked_wide_reduce_n(RandomAccessIterator1 first,
                                       SizeType1 n,
                                       SizeType2 num_blocks,
                                       BinaryFunction binary_op,
                                       RandomAccessIterator2 result)
{
  // XXX this implementation is incredibly ugly
  
  // if we only received one output block, use the standard reduction
  if(num_blocks < 2)
  {
    thrust::detail::device::cuda::unordered_blocked_standard_reduce_n(first, n, num_blocks, binary_op, result);
  } // end if
  else
  {
    // break the reduction into a "wide" body and the "tail"
    // this assumes we have at least two output blocks to work with
    
    // "wide" reduction for small types like char, short, etc.
    typedef typename thrust::iterator_traits<RandomAccessIterator1>::value_type InputType;
    typedef typename thrust::iterator_traits<RandomAccessIterator2>::value_type OutputType;
    typedef unsigned int WideType;

    // note: this assumes that InputIterator is a InputType * and can be reinterpret_casted to WideType *
    
    // TODO use simple threshold and ensure alignment of wide_first
    
    // process first part
    const size_t input_type_per_wide_type = sizeof(WideType) / sizeof(InputType);
    const size_t n_wide = n / input_type_per_wide_type;

    thrust::device_ptr<const WideType> wide_first(reinterpret_cast<const WideType *>(thrust::raw_pointer_cast(&*first)));

    thrust::transform_iterator<
      detail::wide_unary_op<InputType,OutputType,BinaryFunction,WideType>,
      thrust::device_ptr<const WideType>
    > xfrm_wide_first = thrust::make_transform_iterator(wide_first, detail::wide_unary_op<InputType,OutputType,BinaryFunction,WideType>(binary_op));

    // compute where the tail is
    RandomAccessIterator1 tail_first = first + n_wide * input_type_per_wide_type;
    const size_t n_tail = n - (tail_first - first);

    // count the number of results to produce from the widened input
    size_t num_wide_results = num_blocks;
    
    // reserve one of the results for the tail, if there is one
    if(n_tail > 0)
    {
      --num_wide_results;
    }

    // process the wide body
    thrust::detail::device::cuda::unordered_blocked_standard_reduce_n(xfrm_wide_first, n_wide, num_wide_results, binary_op, result);

    // process tail
    thrust::detail::device::cuda::unordered_blocked_standard_reduce_n(tail_first, n_tail, (size_t)1u, binary_op, result + num_wide_results);
  } // end else
} // end unordered_blocked_wide_reduce_n()


template<typename RandomAccessIterator1,
         typename SizeType1,
         typename SizeType2,
         typename BinaryFunction,
         typename RandomAccessIterator2>
  void unordered_blocked_standard_reduce_n(RandomAccessIterator1 first,
                                           SizeType1 n,
                                           SizeType2 num_blocks,
                                           BinaryFunction binary_op,
                                           RandomAccessIterator2 result)
{
  typedef typename thrust::iterator_value<RandomAccessIterator2>::type OutputType;

  // handle zero length input or output
  if(n == 0 || num_blocks == 0)
    return;

  // whether to perform blockwise reductions in shared memory or global memory
  thrust::detail::integral_constant<bool, sizeof(OutputType) <= 64> use_smem;

  return detail::unordered_blocked_reduce_n(first, n, num_blocks, binary_op, result, use_smem);
} // end unordered_standard_blocked_reduce_n()


template<typename RandomAccessIterator1,
         typename SizeType1,
         typename SizeType2,
         typename BinaryFunction,
         typename RandomAccessIterator2>
  void unordered_blocked_reduce_n(RandomAccessIterator1 first,
                                  SizeType1 n,
                                  SizeType2 num_blocks,
                                  BinaryFunction binary_op,
                                  RandomAccessIterator2 result)
{
  // dispatch on whether or not to use a wide reduction
  return thrust::detail::device::cuda::dispatch::unordered_blocked_reduce_n(first, n, num_blocks, binary_op, result,
    typename detail::use_wide_reduction<RandomAccessIterator1>::type());
} // end unordered_blocked_reduce_n()


template<typename RandomAccessIterator,
         typename SizeType,
         typename OutputType,
         typename BinaryFunction>
  SizeType get_unordered_blocked_reduce_n_schedule(RandomAccessIterator first,
                                                   SizeType n,
                                                   OutputType init,
                                                   BinaryFunction binary_op)
{
  // dispatch on whether or not to use the wide reduction
  return thrust::detail::device::cuda::dispatch::get_unordered_blocked_reduce_n_schedule(first, n, init, binary_op,
    typename detail::use_wide_reduction<RandomAccessIterator>::type());
}

} // end namespace cuda
} // end namespace device
} // end namespace detail
} // end namespace thrust

#endif // THRUST_DEVICE_COMPILER != THRUST_DEVICE_COMPILER_NVCC

