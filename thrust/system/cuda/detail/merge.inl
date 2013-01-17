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
#include <thrust/system/cuda/detail/merge.h>
#include <thrust/pair.h>
#include <thrust/tuple.h>
#include <thrust/detail/minmax.h>
#include <thrust/detail/function.h>
#include <thrust/system/cuda/detail/detail/uninitialized.h>
#include <thrust/system/cuda/detail/detail/launch_closure.h>
#include <thrust/detail/util/blocking.h>

namespace thrust
{
namespace system
{
namespace cuda
{
namespace detail
{
namespace merge_detail
{


template<typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename Size,
         typename Compare>
__device__ __thrust_forceinline__
thrust::pair<Size,Size>
  partition_search(RandomAccessIterator1 first1,
                   RandomAccessIterator2 first2,
                   Size diag,
                   Size lower_bound1,
                   Size upper_bound1,
                   Size lower_bound2,
                   Size upper_bound2,
                   Compare comp)
{
  Size begin = thrust::max<Size>(lower_bound1, diag - upper_bound2);
  Size end   = thrust::min<Size>(diag - lower_bound2, upper_bound1);

  while(begin < end)
  {
    Size mid = (begin + end) / 2;
    Size index1 = mid;
    Size index2 = diag - mid - 1;

    if(comp(first2[index2], first1[index1]))
    {
      end = mid;
    }
    else
    {
      begin = mid + 1;
    }
  }

  return thrust::make_pair(begin, diag - begin);
}


template<typename Context, typename RandomAccessIterator1, typename Size, typename RandomAccessIterator2, typename RandomAccessIterator3, typename Compare>
__device__ __thrust_forceinline__
void merge_n(Context &ctx,
             RandomAccessIterator1 first1,
             Size n1,
             RandomAccessIterator2 first2,
             Size n2,
             RandomAccessIterator3 result,
             Compare comp_,
             unsigned int work_per_thread)
{
  const unsigned int block_size = ctx.block_dimension();
  thrust::detail::device_function<Compare,bool> comp(comp_);
  typedef typename thrust::iterator_value<RandomAccessIterator1>::type value_type1;
  typedef typename thrust::iterator_value<RandomAccessIterator2>::type value_type2;

  Size result_size = n1 + n2;

  // this is just oversubscription_rate * block_size * work_per_thread
  // but it makes no sense to send oversubscription_rate as an extra parameter
  Size work_per_block = thrust::detail::util::divide_ri(result_size, ctx.grid_dimension());

  using thrust::system::cuda::detail::detail::uninitialized;
  __shared__ uninitialized<thrust::pair<Size,Size> > s_block_input_begin;

  Size result_block_offset = ctx.block_index() * work_per_block;

  // find where this block's input begins in both input sequences
  if(ctx.thread_index() == 0)
  {
    s_block_input_begin = (ctx.block_index() == 0) ?
      thrust::pair<Size,Size>(0,0) :
      partition_search(first1, first2,
                       result_block_offset,
                       Size(0), n1,
                       Size(0), n2,
                       comp);
  }

  ctx.barrier();

  // iterate to consume this block's input
  Size work_per_iteration = block_size * work_per_thread;
  thrust::pair<Size,Size> block_input_end = s_block_input_begin;
  block_input_end.first  += work_per_iteration;
  block_input_end.second += work_per_iteration;
  Size result_block_offset_last = result_block_offset + thrust::min<Size>(work_per_block, result_size - result_block_offset);

  for(;
      result_block_offset < result_block_offset_last;
      result_block_offset += work_per_iteration,
      block_input_end.first  += work_per_iteration,
      block_input_end.second += work_per_iteration
     )
  {
    // find where this thread's input begins in both input sequences for this iteration
    thrust::pair<Size,Size> thread_input_begin =
      partition_search(first1, first2,
                       Size(result_block_offset + ctx.thread_index() * work_per_thread),
                       s_block_input_begin.get().first,  thrust::min<Size>(block_input_end.first , n1),
                       s_block_input_begin.get().second, thrust::min<Size>(block_input_end.second, n2),
                       comp);

    ctx.barrier();

    // XXX the performance impact of not keeping x1 & x2
    //     in registers is about 10% for int32
    uninitialized<value_type1> x1;
    uninitialized<value_type2> x2;

    // XXX this is just a serial merge -- try to simplify or abstract this loop
    Size i = result_block_offset + ctx.thread_index() * work_per_thread;
    Size last_i = i + thrust::min<Size>(work_per_thread, result_size - thread_input_begin.first - thread_input_begin.second);
    for(;
        i < last_i;
        ++i)
    {
      // optionally load x1 & x2
      bool output_x2 = true;
      if(thread_input_begin.second < n2)
      {
        x2 = first2[thread_input_begin.second];
      }
      else
      {
        output_x2 = false;
      }

      if(thread_input_begin.first < n1)
      {
        x1 = first1[thread_input_begin.first];

        if(output_x2)
        {
          output_x2 = comp(x2.get(), x1.get());
        }
      }

      result[i] = output_x2 ? x2.get() : x1.get();

      if(output_x2)
      {
        ++thread_input_begin.second;
      }
      else
      {
        ++thread_input_begin.first;
      }
    } // end for

    // the block's last thread has conveniently located the
    // beginning of the next iteration's input
    if(ctx.thread_index() == block_size-1)
    {
      s_block_input_begin = thread_input_begin;
    }
    ctx.barrier();
  } // end for
} // end merge_n


template<typename RandomAccessIterator1, typename Size, typename RandomAccessIterator2, typename RandomAccessIterator3, typename Compare>
  struct merge_n_closure
{
  typedef thrust::system::cuda::detail::detail::blocked_thread_array context_type;

  RandomAccessIterator1 first1;
  Size n1;
  RandomAccessIterator2 first2;
  Size n2;
  RandomAccessIterator3 result;
  Compare comp;
  Size work_per_thread;

  merge_n_closure(RandomAccessIterator1 first1, Size n1, RandomAccessIterator2 first2, Size n2, RandomAccessIterator3 result, Compare comp, Size work_per_thread)
    : first1(first1), n1(n1), first2(first2), n2(n2), result(result), comp(comp), work_per_thread(work_per_thread)
  {}

  __device__ __forceinline__
  void operator()()
  {
    context_type ctx;
    merge_n(ctx, first1, n1, first2, n2, result, comp, work_per_thread);
  }
};


// returns (work_per_thread, threads_per_block, oversubscription_factor)
template<typename RandomAccessIterator1, typename RandomAccessIterator2, typename RandomAccessIterator3, typename Compare>
  thrust::tuple<unsigned int,unsigned int,unsigned int>
    tunables(RandomAccessIterator1, RandomAccessIterator1, RandomAccessIterator2, RandomAccessIterator2, RandomAccessIterator3, Compare comp)
{
  // determined by empirical testing on GTX 480
  // ~4500 Mkeys/s on GTX 480
  const unsigned int work_per_thread         = 5;
  const unsigned int threads_per_block       = 128;
  const unsigned int oversubscription_factor = 30;

  return thrust::make_tuple(work_per_thread, threads_per_block, oversubscription_factor);
}


} // end merge_detail


template<typename System,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2, 
	 typename RandomAccessIterator3,
         typename Compare>
RandomAccessIterator3 merge(dispatchable<System> &system,
                            RandomAccessIterator1 first1,
                            RandomAccessIterator1 last1,
                            RandomAccessIterator2 first2,
                            RandomAccessIterator2 last2,
                            RandomAccessIterator3 result,
                            Compare comp)
{
  typedef typename thrust::iterator_difference<RandomAccessIterator1>::type Size;
  Size n1 = last1 - first1;
  Size n2 = last2 - first2;
  typename thrust::iterator_difference<RandomAccessIterator1>::type n = n1 + n2;

  // empty result
  if(n <= 0) return result;

  unsigned int work_per_thread = 0, threads_per_block = 0, oversubscription_factor = 0;
  thrust::tie(work_per_thread,threads_per_block,oversubscription_factor)
    = merge_detail::tunables(first1, last1, first2, last2, result, comp);

  const unsigned int work_per_block = work_per_thread * threads_per_block;

  const unsigned int num_processors = device_properties().multiProcessorCount;
  const unsigned int num_blocks = thrust::min<int>(oversubscription_factor * num_processors, thrust::detail::util::divide_ri(n, work_per_block));

  typedef merge_detail::merge_n_closure<RandomAccessIterator1,Size,RandomAccessIterator2,RandomAccessIterator3,Compare> closure_type;
  closure_type closure(first1, n1, first2, n2, result, comp, work_per_thread);

  detail::launch_closure(closure, num_blocks, threads_per_block);

  return result + n1 + n2;
} // end merge()


} // end namespace detail
} // end namespace cuda
} // end namespace system
} // end namespace thrust

