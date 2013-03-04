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

#include <thrust/detail/config.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/detail/minmax.h>
#include <thrust/detail/temporary_array.h>
#include <thrust/detail/internal_functional.h>
#include <thrust/system/detail/internal/decompose.h>
#include <thrust/scan.h>
#include <thrust/system/cuda/detail/default_decomposition.h>
#include <thrust/system/cuda/detail/reduce_intervals.h>
#include <thrust/system/cuda/detail/block/inclusive_scan.h>
#include <thrust/system/cuda/detail/detail/launch_closure.h>
#include <thrust/system/cuda/detail/execution_policy.h>
#include <thrust/functional.h>


__THRUST_DISABLE_MSVC_POSSIBLE_LOSS_OF_DATA_WARNING_BEGIN

namespace thrust
{
namespace system
{
namespace cuda
{
namespace detail
{

template <typename InputIterator1,
          typename InputIterator2,
          typename InputIterator3,
          typename Decomposition,
          typename OutputIterator,
          typename Context>
struct copy_if_intervals_closure
{
  InputIterator1 input;
  InputIterator2 stencil;
  InputIterator3 offsets;
  Decomposition decomp;
  OutputIterator output;

  typedef Context context_type;
  context_type context;
  
  copy_if_intervals_closure(InputIterator1 input,
                            InputIterator2 stencil,
                            InputIterator3 offsets,
                            Decomposition decomp,
                            OutputIterator output,
                            Context context = Context())
    : input(input), stencil(stencil), offsets(offsets), decomp(decomp), output(output), context(context) {}

  __device__ __thrust_forceinline__
  void operator()(void)
  {
    typedef typename thrust::iterator_value<OutputIterator>::type OutputType;
   
    typedef unsigned int PredicateType;
    
    const unsigned int CTA_SIZE = context_type::ThreadsPerBlock::value;

    thrust::plus<PredicateType> binary_op;

    __shared__ PredicateType sdata[CTA_SIZE];  context.barrier();
    
    typedef typename Decomposition::index_type IndexType;

    // this block processes results in [range.begin(), range.end())
    thrust::system::detail::internal::index_range<IndexType> range = decomp[context.block_index()];

    IndexType base = range.begin();

    PredicateType predicate = 0;
    
    // advance input iterators to this thread's starting position
    input   += base + context.thread_index();
    stencil += base + context.thread_index();

    // advance output to this interval's starting position
    if (context.block_index() != 0)
    {
        InputIterator3 temp = offsets + (context.block_index() - 1);
        output += *temp;
    }

    // process full blocks
    while(base + CTA_SIZE <= range.end())
    {
        // read data
        sdata[context.thread_index()] = predicate = *stencil;
      
        context.barrier();

        // scan block
        block::inclusive_scan(context, sdata, binary_op);
       
        // write data
        if (predicate)
        {
            OutputIterator temp2 = output + (sdata[context.thread_index()] - 1);
            *temp2 = *input;
        }

        // advance inputs by CTA_SIZE
        base    += CTA_SIZE;
        input   += CTA_SIZE;
        stencil += CTA_SIZE;

        // advance output by number of true predicates
        output += sdata[CTA_SIZE - 1];

        context.barrier();
    }

    // process partially full block at end of input (if necessary)
    if (base < range.end())
    {
        // read data
        if (base + context.thread_index() < range.end())
            sdata[context.thread_index()] = predicate = *stencil;
        else
            sdata[context.thread_index()] = predicate = 0;
       
        context.barrier();

        // scan block
        block::inclusive_scan(context, sdata, binary_op);
       
        // write data
        if (predicate) // expects predicate=false for >= interval_end
        {
            OutputIterator temp2 = output + (sdata[context.thread_index()] - 1);
            *temp2 = *input;
        }
    }
  }
}; // copy_if_intervals_closure


template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename Predicate>
   OutputIterator copy_if(execution_policy<DerivedPolicy> &exec,
                          InputIterator1 first,
                          InputIterator1 last,
                          InputIterator2 stencil,
                          OutputIterator output,
                          Predicate pred)
{
  typedef typename thrust::iterator_difference<InputIterator1>::type IndexType;
  typedef typename thrust::iterator_value<OutputIterator>::type      OutputType;

  if (first == last)
      return output;

  typedef thrust::system::detail::internal::uniform_decomposition<IndexType> Decomposition;
  typedef thrust::detail::temporary_array<IndexType, DerivedPolicy>          IndexArray;

  Decomposition decomp = default_decomposition(last - first);

  // storage for per-block predicate counts
  IndexArray block_results(exec, decomp.size());

  // convert stencil into an iterator that produces integral values in {0,1}
  typedef typename thrust::detail::predicate_to_integral<Predicate,IndexType>              PredicateToIndexTransform;
  typedef thrust::transform_iterator<PredicateToIndexTransform, InputIterator2, IndexType> PredicateToIndexIterator;

  PredicateToIndexIterator predicate_stencil(stencil, PredicateToIndexTransform(pred));

  // compute number of true values in each interval
  thrust::system::cuda::detail::reduce_intervals(exec, predicate_stencil, block_results.begin(), thrust::plus<IndexType>(), decomp);

  // scan the partial sums
  thrust::inclusive_scan(exec, block_results.begin(), block_results.end(), block_results.begin(), thrust::plus<IndexType>());

  // copy values to output
  const unsigned int ThreadsPerBlock = 256;
  typedef typename IndexArray::iterator InputIterator3;
  typedef detail::statically_blocked_thread_array<ThreadsPerBlock> Context;
  typedef copy_if_intervals_closure<InputIterator1,PredicateToIndexIterator,InputIterator3,Decomposition,OutputIterator,Context> Closure;
  Closure closure(first, predicate_stencil, block_results.begin(), decomp, output);
  detail::launch_closure(closure, decomp.size(), ThreadsPerBlock);

  return output + block_results[decomp.size() - 1];
} // end copy_if()


} // end namespace detail
} // end namespace cuda
} // end namespace system
} // end namespace thrust

__THRUST_DISABLE_MSVC_POSSIBLE_LOSS_OF_DATA_WARNING_END

