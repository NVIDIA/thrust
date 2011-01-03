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
#include <thrust/detail/device/cuda/synchronize.h>
#include <thrust/detail/device/cuda/arch.h>


__THRUST_DISABLE_MSVC_POSSIBLE_LOSS_OF_DATA_WARNING_BEGIN

namespace thrust
{
namespace detail
{

// XXX WAR circular inclusion problem with this forward declaration
template <typename> class raw_cuda_device_buffer;

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
   
    typedef unsigned int PredicateType;

    thrust::plus<PredicateType> binary_op;

    __shared__ PredicateType sdata[CTA_SIZE];  __syncthreads();

    const IndexType interval_begin = interval_size * blockIdx.x;
    const IndexType interval_end   = min(interval_begin + interval_size, N);

    IndexType base = interval_begin;

    PredicateType predicate = 0;
    
    // advance input iterators to this thread's starting position
    input   += interval_begin + threadIdx.x;
    stencil += interval_begin + threadIdx.x;

    // advance output to this interval's starting position
    if (blockIdx.x != 0)
    {
        InputIterator3 temp = offsets + (blockIdx.x - 1);
        output += thrust::detail::device::dereference(temp);
    }

    // process full blocks
    while(base + CTA_SIZE <= interval_end)
    {
        // read data
        sdata[threadIdx.x] = predicate = thrust::detail::device::dereference(stencil);
       
        __syncthreads();

        // scan block
        block::inplace_inclusive_scan<CTA_SIZE>(sdata, binary_op);
       
        // write data
        if (predicate)
        {
            OutputIterator temp2 = output + (sdata[threadIdx.x] - 1);
            thrust::detail::device::dereference(temp2) = thrust::detail::device::dereference(input);
        }

        // advance inputs by CTA_SIZE
        base    += CTA_SIZE;
        input   += CTA_SIZE;
        stencil += CTA_SIZE;

        // advance output by number of true predicates
        output += sdata[CTA_SIZE - 1];

        __syncthreads();
    }

    // process partially full block at end of input (if necessary)
    if (base < interval_end)
    {
        // read data
        if (base + threadIdx.x < interval_end)
            sdata[threadIdx.x] = predicate = thrust::detail::device::dereference(stencil);
        else
            sdata[threadIdx.x] = predicate = 0;
        
        __syncthreads();

        // scan block
        block::inplace_inclusive_scan<CTA_SIZE>(sdata, binary_op);
       
        // write data
        if (predicate) // expects predicate=false for >= interval_end
        {
            OutputIterator temp2 = output + (sdata[threadIdx.x] - 1);
            thrust::detail::device::dereference(temp2) = thrust::detail::device::dereference(input);
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
    const IndexType max_intervals = 3 * (thrust::detail::device::cuda::arch::max_active_threads() / CTA_SIZE);  // TODO put this in a common place

    thrust::pair<IndexType, IndexType> splitting = uniform_interval_splitting<IndexType>(N, 32, max_intervals);

    const IndexType interval_size = splitting.first;
    const IndexType num_intervals = splitting.second;

    thrust::detail::raw_cuda_device_buffer<IndexType> block_results(num_intervals);

    // convert stencil into an iterator that produces integral values in {0,1}
    typedef typename thrust::detail::predicate_to_integral<Predicate,IndexType>              PredicateToIndexTransform;
    typedef thrust::transform_iterator<PredicateToIndexTransform, InputIterator2, IndexType> PredicateToIndexIterator;

    PredicateToIndexIterator predicate_stencil(stencil, PredicateToIndexTransform(pred));

    reduce_intervals<CTA_SIZE> <<<num_intervals, CTA_SIZE>>>
        (predicate_stencil,
         N,
         interval_size,
         block_results.begin(),
         thrust::plus<IndexType>());
    synchronize_if_enabled("reduce_intervals");

    thrust::detail::device::inclusive_scan(block_results.begin(), block_results.end(), block_results.begin(), thrust::plus<IndexType>());

    copy_if_intervals<CTA_SIZE> <<<num_intervals, CTA_SIZE>>>
        (first,
         predicate_stencil,
         block_results.begin(),
         N,
         interval_size,
         output);
    synchronize_if_enabled("copy_if_intervals");

    return output + block_results[num_intervals - 1];
}

} // end namespace cuda
} // end namespace device
} // end namespace detail
} // end namespace thrust

__THRUST_DISABLE_MSVC_POSSIBLE_LOSS_OF_DATA_WARNING_END

#endif // THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC


