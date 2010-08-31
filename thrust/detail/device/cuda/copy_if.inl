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


// do not attempt to compile this file with any other compiler
#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC

#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/transform_iterator.h>

#include <thrust/detail/util/blocking.h>

#include <thrust/detail/raw_buffer.h>
#include <thrust/detail/internal_functional.h>
#include <thrust/detail/device/dereference.h>

#include <thrust/detail/device/cuda/partition.h>
#include <thrust/detail/device/cuda/block/reduce.h>
#include <thrust/detail/device/cuda/block/inclusive_scan.h>


#if THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_MSVC
// temporarily disable 'possible loss of data' warnings on MSVC
#pragma warning(push)
#pragma warning(disable : 4244 4267)
#endif

namespace thrust
{
namespace detail
{
namespace device
{
namespace cuda
{


template <unsigned int CTA_SIZE,
          typename InputIterator,
          typename IndexType,
          typename OutputIterator,
          typename BinaryFunction>
__launch_bounds__(CTA_SIZE,1)
__global__
void reduce_intervals(InputIterator input,
                      const IndexType n,
                      const IndexType interval_size,
                      OutputIterator output,
                      BinaryFunction binary_op)
{
    typedef typename thrust::iterator_value<OutputIterator>::type OutputType;

    __shared__ OutputType shared_array[CTA_SIZE];

    const IndexType interval_begin = interval_size * blockIdx.x;
    const IndexType interval_end   = min(interval_begin + interval_size, n);

    IndexType i = interval_begin + threadIdx.x;

    // advance input
    input += i;

    if (i < interval_end)
    {
        // initialize local sum 
        OutputType sum = thrust::detail::device::dereference(input);

        i     += CTA_SIZE;
        input += CTA_SIZE;

        // accumulate local sum
        while (i < interval_end)
        {
            OutputType val = thrust::detail::device::dereference(input);
            sum = binary_op(sum, val);

            i     += CTA_SIZE;
            input += CTA_SIZE;
        }

        // copy local sum to shared memory
        shared_array[threadIdx.x] = sum;
    }

    __syncthreads();

    // compute reduction across block
    thrust::detail::device::cuda::block::reduce_n(shared_array, min(interval_end - interval_begin, IndexType(CTA_SIZE)), binary_op);

    // write result for this block to global mem 
    if (threadIdx.x == 0) 
    {
        output += blockIdx.x;
        thrust::detail::device::dereference(output) = shared_array[0];
    }
} // end reduce_n_device()

template <unsigned int CTA_SIZE,
          typename InputIterator1,
          typename InputIterator2,
          typename InputIterator3,
          typename IndexType,
          typename OutputIterator>
__launch_bounds__(CTA_SIZE,1)
__global__
void copy_if_intervals(InputIterator1 input,
                       InputIterator2 stencil,
                       InputIterator3 offsets,
                       const IndexType N,
                       const IndexType interval_size,
                       OutputIterator output)
{
    typedef typename thrust::iterator_value<OutputIterator>::type OutputType;
    
    thrust::plus<IndexType> binary_op;

    __shared__ IndexType sdata[CTA_SIZE];  __syncthreads();

    const IndexType interval_begin = interval_size * blockIdx.x;
    const IndexType interval_end   = min(interval_begin + interval_size, N);

    IndexType base = interval_begin;

    IndexType predicate = 0;

    // initial offset
    if (threadIdx.x == 0)
    {
        if (blockIdx.x == 0)
        {
            sdata[CTA_SIZE - 1] = 0;
        }
        else
        {
            InputIterator3 temp = offsets + (blockIdx.x - 1);
            sdata[CTA_SIZE - 1] = thrust::detail::device::dereference(temp);
        }
    }

    // process full blocks
    for(; base + CTA_SIZE <= interval_end; base += CTA_SIZE)
    {
        // read data
        {
            InputIterator2 temp = stencil + (base + threadIdx.x);
            predicate = thrust::detail::device::dereference(temp);
        }
       
        // carry in
        if (threadIdx.x == 0)
        {
            IndexType tmp = sdata[CTA_SIZE - 1];
            sdata[0] = binary_op(tmp, predicate);
        }

        __syncthreads();

        if (threadIdx.x != 0)
            sdata[threadIdx.x] = predicate;

        __syncthreads();


        // scan block
        block::inplace_inclusive_scan<CTA_SIZE>(sdata, binary_op);
       
        // write data
        if (predicate)
        {
            InputIterator1 temp1 = input  + (base + threadIdx.x);
            OutputIterator temp2 = output + (sdata[threadIdx.x] - 1);
            thrust::detail::device::dereference(temp2) = thrust::detail::device::dereference(temp1);
        }
    }

    // process partially full block at end of input (if necessary)
    if (base < interval_end)
    {
        // read data
        if (base + threadIdx.x < interval_end)
        {
            InputIterator2 temp = stencil + (base + threadIdx.x);
            predicate = thrust::detail::device::dereference(temp);
        }
        else
        {
            predicate = 0;
        }
        
        // carry in
        if (threadIdx.x == 0)
        {
            IndexType tmp = sdata[CTA_SIZE - 1];
            sdata[0] = binary_op(tmp, predicate);
        }

        __syncthreads();

        if (threadIdx.x != 0)
            sdata[threadIdx.x] = predicate;

        __syncthreads();

        // scan block
        block::inplace_inclusive_scan<CTA_SIZE>(sdata, binary_op);
       
        // write data
        if (predicate) // expects predicate=false for >= interval_end
        {
            InputIterator1 temp1 = input  + (base + threadIdx.x);
            OutputIterator temp2 = output + (sdata[threadIdx.x] - 1);
            thrust::detail::device::dereference(temp2) = thrust::detail::device::dereference(temp1);
        }
    }
}



template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename Predicate>
   OutputIterator copy_if(InputIterator1 first,
                          InputIterator1 last,
                          InputIterator2 stencil,
                          OutputIterator output,
                          Predicate pred)
{
    typedef typename thrust::iterator_difference<InputIterator1>::type IndexType;
    typedef typename thrust::iterator_value<OutputIterator>::type      OutputType;

    if (first == last)
        return output;

    const IndexType CTA_SIZE      = 256;
    const IndexType N             = last - first;
    const IndexType max_intervals = 3 * (thrust::experimental::arch::max_active_threads() / CTA_SIZE);

    thrust::pair<IndexType, IndexType> splitting = uniform_interval_splitting<IndexType>(N, 32, max_intervals);

    const IndexType interval_size = splitting.first;
    const IndexType num_intervals = splitting.second;

    thrust::detail::raw_cuda_device_buffer<IndexType> block_results(num_intervals);

    typedef typename thrust::detail::predicate_to_integral<Predicate,IndexType>              PredicateToIndexTransform;
    typedef thrust::transform_iterator<PredicateToIndexTransform, InputIterator2, IndexType> PredicateToIndexIterator;

    PredicateToIndexIterator predicate_stencil(stencil, PredicateToIndexTransform(pred));

    reduce_intervals<CTA_SIZE> <<<num_intervals, CTA_SIZE>>>
        (predicate_stencil,
         N,
         interval_size,
         block_results.begin(),
         thrust::plus<IndexType>());

    thrust::detail::device::inclusive_scan(block_results.begin(), block_results.end(), block_results.begin(), thrust::plus<IndexType>());

    copy_if_intervals<CTA_SIZE> <<<num_intervals, CTA_SIZE>>>
        (first,
         predicate_stencil,
         block_results.begin(),
         N,
         interval_size,
         output);

    return output + block_results[num_intervals - 1];
}

} // end namespace cuda
} // end namespace device
} // end namespace detail
} // end namespace thrust


#if THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_MSVC
// reenable 'possible loss of data' warnings
#pragma warning(pop)
#endif

#endif // THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC


