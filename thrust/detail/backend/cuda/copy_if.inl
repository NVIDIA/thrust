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


// do not attempt to compile this file with any other compiler
#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC

#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/transform_iterator.h>

#include <thrust/detail/minmax.h>
#include <thrust/detail/uninitialized_array.h>
#include <thrust/detail/internal_functional.h>
#include <thrust/detail/util/blocking.h>

#include <thrust/detail/backend/dereference.h>
#include <thrust/detail/backend/decompose.h>
#include <thrust/detail/backend/reduce_intervals.h>

#include <thrust/detail/backend/cuda/synchronize.h>
#include <thrust/detail/backend/cuda/default_decomposition.h>
#include <thrust/detail/backend/cuda/block/reduce.h>
#include <thrust/detail/backend/cuda/block/inclusive_scan.h>


__THRUST_DISABLE_MSVC_POSSIBLE_LOSS_OF_DATA_WARNING_BEGIN

namespace thrust
{
namespace detail
{

// XXX WAR circular inclusion problem with this forward declaration
template <typename,typename> class uninitialized_array;

namespace backend
{
namespace cuda
{

template <unsigned int CTA_SIZE,
          typename InputIterator1,
          typename InputIterator2,
          typename InputIterator3,
          typename Decomposition,
          typename OutputIterator>
__launch_bounds__(CTA_SIZE,1)
__global__
void copy_if_intervals(InputIterator1 input,
                       InputIterator2 stencil,
                       InputIterator3 offsets,
                       Decomposition decomp,
                       OutputIterator output)
{
    typedef typename thrust::iterator_value<OutputIterator>::type OutputType;
   
    typedef unsigned int PredicateType;

    thrust::plus<PredicateType> binary_op;

    __shared__ PredicateType sdata[CTA_SIZE];  __syncthreads();
    
    typedef typename Decomposition::index_type IndexType;

    // this block processes results in [range.begin(), range.end())
    thrust::detail::backend::index_range<IndexType> range = decomp[blockIdx.x];

    IndexType base = range.begin();

    PredicateType predicate = 0;
    
    // advance input iterators to this thread's starting position
    input   += base + threadIdx.x;
    stencil += base + threadIdx.x;

    // advance output to this interval's starting position
    if (blockIdx.x != 0)
    {
        InputIterator3 temp = offsets + (blockIdx.x - 1);
        output += thrust::detail::backend::dereference(temp);
    }

    // process full blocks
    while(base + CTA_SIZE <= range.end())
    {
        // read data
        sdata[threadIdx.x] = predicate = thrust::detail::backend::dereference(stencil);
       
        __syncthreads();

        // scan block
        block::inplace_inclusive_scan<CTA_SIZE>(sdata, binary_op);
       
        // write data
        if (predicate)
        {
            OutputIterator temp2 = output + (sdata[threadIdx.x] - 1);
            thrust::detail::backend::dereference(temp2) = thrust::detail::backend::dereference(input);
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
    if (base < range.end())
    {
        // read data
        if (base + threadIdx.x < range.end())
            sdata[threadIdx.x] = predicate = thrust::detail::backend::dereference(stencil);
        else
            sdata[threadIdx.x] = predicate = 0;
        
        __syncthreads();

        // scan block
        block::inplace_inclusive_scan<CTA_SIZE>(sdata, binary_op);
       
        // write data
        if (predicate) // expects predicate=false for >= interval_end
        {
            OutputIterator temp2 = output + (sdata[threadIdx.x] - 1);
            thrust::detail::backend::dereference(temp2) = thrust::detail::backend::dereference(input);
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

  thrust::detail::backend::uniform_decomposition<IndexType> decomp = thrust::detail::backend::cuda::default_decomposition(last - first);

  thrust::detail::uninitialized_array<IndexType, thrust::detail::cuda_device_space_tag> block_results(decomp.size());

  // convert stencil into an iterator that produces integral values in {0,1}
  typedef typename thrust::detail::predicate_to_integral<Predicate,IndexType>              PredicateToIndexTransform;
  typedef thrust::transform_iterator<PredicateToIndexTransform, InputIterator2, IndexType> PredicateToIndexIterator;

  PredicateToIndexIterator predicate_stencil(stencil, PredicateToIndexTransform(pred));

  // compute number of true values in each interval
  thrust::detail::backend::cuda::reduce_intervals(predicate_stencil, block_results.begin(), thrust::plus<IndexType>(), decomp);

  // scan the partial sums
  thrust::detail::backend::inclusive_scan(block_results.begin(), block_results.end(), block_results.begin(), thrust::plus<IndexType>());

  copy_if_intervals<256> <<<decomp.size(), 256>>>
      (first,
       predicate_stencil,
       block_results.begin(),
       decomp,
       output);
  synchronize_if_enabled("copy_if_intervals");

  return output + block_results[decomp.size() - 1];
}

} // end namespace cuda
} // end namespace backend
} // end namespace detail
} // end namespace thrust

__THRUST_DISABLE_MSVC_POSSIBLE_LOSS_OF_DATA_WARNING_END

#endif // THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC

