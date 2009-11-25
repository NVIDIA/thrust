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

#include <thrust/functional.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/transform_iterator.h>

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
template<unsigned int block_size,
         typename InputIterator,
         typename OutputType,
         typename BinaryFunction>
  __global__ void
  __thrust__unordered_reduce_kernel(InputIterator input,
                                    const unsigned int n,
                                    OutputType * block_results,  
                                    BinaryFunction binary_op)
{
    __shared__ unsigned char sdata_workaround[block_size * sizeof(OutputType)];
    OutputType *sdata = reinterpret_cast<OutputType*>(sdata_workaround);

    // perform first level of reduction,
    // write per-block results to global memory for second level reduction
    
    const unsigned int grid_size = block_size * gridDim.x;
    unsigned int i = blockIdx.x * block_size + threadIdx.x;

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
    thrust::detail::block::reduce_n<block_size>(sdata, n, binary_op);

    // write result for this block to global mem 
    if (threadIdx.x == 0) 
        block_results[blockIdx.x] = sdata[threadIdx.x];

} // end __thrust__unordered_reduce_kernel()


template<typename InputIterator,
         typename OutputType,
         typename BinaryFunction>
  OutputType reduce_n(InputIterator input,
                      const size_t n,
                      OutputType init,
                      BinaryFunction binary_op)
{
    // handle zero length array case first
    if( n == 0 )
        return init;

    // 16KB (max) - 1KB (upper bound on what's used for other purposes)
    const size_t max_smem_size = 15 * 1024; 

    // largest 2^N that fits in SMEM
    static const size_t blocksize_limit1 = 1 << thrust::detail::mpl::math::log2< (max_smem_size/sizeof(OutputType)) >::value;
    static const size_t blocksize_limit2 = 256;

    static const size_t block_size = (blocksize_limit1 < blocksize_limit2) ? blocksize_limit1 : blocksize_limit2;
    
    const size_t max_blocks = thrust::experimental::arch::max_active_blocks(__thrust__unordered_reduce_kernel<block_size, InputIterator, OutputType, BinaryFunction>, block_size, (size_t) 0);

    const unsigned int grid_size = std::max((size_t) 1, std::min( (n / block_size), max_blocks));

    // allocate storage for per-block results
    thrust::detail::raw_device_buffer<OutputType> temp(grid_size + 1);

    // set first element of temp array to init
    temp[0] = init;

    // reduce input to per-block sums
    __thrust__unordered_reduce_kernel<block_size>
        <<<grid_size, block_size>>>(input, n, raw_pointer_cast(&temp[1]), binary_op);

    // reduce per-block sums together with init
    __thrust__unordered_reduce_kernel<block_size>
        <<<1, block_size>>>(raw_pointer_cast(&temp[0]), grid_size + 1, raw_pointer_cast(&temp[0]), binary_op);

    return temp[0];
} // end reduce_n()



namespace detail
{
//////////////    
// Functors //
//////////////    

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


template<typename InputIterator, 
         typename OutputType,
         typename BinaryFunction>
  OutputType reduce_device(InputIterator first,
                           InputIterator last,
                           OutputType init,
                           BinaryFunction binary_op,
                           thrust::detail::true_type)
{
    // "wide" reduction for small types like char, short, etc.
    typedef typename thrust::iterator_traits<InputIterator>::value_type InputType;
    typedef unsigned int WideType;

    // note: this assumes that InputIterator is a InputType * and can be reinterpret_casted to WideType *
   
    // TODO use simple threshold and ensure alignment of wide_first

    // process first part
    size_t input_type_per_wide_type = sizeof(WideType) / sizeof(InputType);
    size_t n_wide = (last - first) / input_type_per_wide_type;

    WideType * wide_first = reinterpret_cast<WideType *>(thrust::raw_pointer_cast(&*first));

    OutputType result = thrust::detail::device::cuda::reduce_n
        (thrust::make_transform_iterator(wide_first, wide_unary_op<InputType,OutputType,BinaryFunction,WideType>(binary_op)),
         n_wide, init, binary_op);

    // process tail
    InputIterator tail_first = first + n_wide * input_type_per_wide_type;
    return thrust::detail::device::cuda::reduce_n(tail_first, last - tail_first, result, binary_op);
    
//    // process first part
//    size_t input_type_per_wide_type = sizeof(WideType) / sizeof(InputType);
//    size_t n_wide = (last - first) / input_type_per_wide_type;
//    wide_reduce_functor<InputType, OutputType, BinaryFunction, WideType> wide_func((&*first).get(), binary_op);
//    OutputType result = thrust::detail::device::cuda::reduce_n(wide_func, n_wide, init, binary_op);
//
//    // process tail
//    InputIterator tail_first = first + n_wide * input_type_per_wide_type;
//    reduce_functor<InputIterator, OutputType> tail_func(tail_first);
//    return thrust::detail::device::cuda::reduce_n(tail_func, last - tail_first, result, binary_op);
}

template<typename InputIterator, 
         typename OutputType,
         typename BinaryFunction>
  OutputType reduce_device(InputIterator first,
                           InputIterator last,
                           OutputType init,
                           BinaryFunction binary_op,
                           thrust::detail::false_type)
{
    // standard reduction
    return thrust::detail::device::cuda::reduce_n(first, last - first, init, binary_op);
}

} // end namespace detail


template<typename InputIterator, 
         typename OutputType,
         typename BinaryFunction>
  OutputType reduce(InputIterator first,
                    InputIterator last,
                    OutputType init,
                    BinaryFunction binary_op)
{
    typedef typename thrust::iterator_traits<InputIterator>::value_type InputType;

    const bool use_wide_load = thrust::detail::is_pod<InputType>::value 
                                    && thrust::detail::is_trivial_iterator<InputIterator>::value
                                    && (sizeof(InputType) == 1 || sizeof(InputType) == 2);
                                    
    return detail::reduce_device(first, last, init, binary_op, thrust::detail::integral_constant<bool, use_wide_load>());
}

} // end namespace cuda
} // end namespace device
} // end namespace detail
} // end namespace thrust

#endif // __CUDACC__

