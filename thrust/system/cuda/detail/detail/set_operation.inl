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
#include <thrust/system/cuda/detail/detail/set_operation.h>
#include <thrust/system/cuda/detail/detail/balanced_path.h>
#include <thrust/system/cuda/detail/block/inclusive_scan.h>
#include <thrust/system/cuda/detail/block/exclusive_scan.h>
#include <thrust/system/cuda/detail/block/copy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>
#include <thrust/scan.h>
#include <thrust/pair.h>
#include <thrust/detail/util/blocking.h>
#include <thrust/detail/temporary_array.h>
#include <thrust/detail/cstdint.h>
#include <thrust/detail/minmax.h>


namespace thrust
{
namespace system
{
namespace cuda
{
namespace detail
{
namespace detail
{
namespace set_operation_detail
{


using thrust::system::cuda::detail::detail::statically_blocked_thread_array;
using thrust::detail::uint16_t;
using thrust::detail::uint32_t;


// empirically determined on sm_20
// value_types larger than this will fail to launch if placed in smem
template<typename T>
  struct stage_through_smem
{
  static const bool value = sizeof(T) <= 6 * sizeof(uint32_t);
};


// max_input_size <= 32
template<typename Size, typename InputIterator, typename OutputIterator>
inline __device__
  OutputIterator serial_bounded_copy_if(Size max_input_size,
                                        InputIterator first,
                                        uint32_t mask,
                                        OutputIterator result)
{
  for(Size i = 0; i < max_input_size; ++i, ++first)
  {
    if((1<<i) & mask)
    {
      *result = *first;
      ++result;
    }
  }

  return result;
}


template<typename Size, typename InputIterator1, typename InputIterator2, typename Compare>
  struct find_partition_offsets_functor
{
  Size partition_size;
  InputIterator1 first1;
  InputIterator2 first2;
  Size n1, n2;
  Compare comp;

  find_partition_offsets_functor(Size partition_size,
                                 InputIterator1 first1, InputIterator1 last1,
                                 InputIterator2 first2, InputIterator2 last2,
                                 Compare comp)
    : partition_size(partition_size),
      first1(first1), first2(first2),
      n1(last1 - first1), n2(last2 - first2),
      comp(comp)
  {}

  inline __host__ __device__
  thrust::pair<Size,Size> operator()(Size i) const
  {
    Size diag = thrust::min(n1 + n2, i * partition_size);

    // XXX the correctness of balanced_path depends critically on the ll suffix below
    //     why???
    return balanced_path(first1, n1, first2, n2, diag, 4ll, comp);
  }
};


template<typename Size, typename System, typename InputIterator1, typename InputIterator2, typename OutputIterator, typename Compare>
  OutputIterator find_partition_offsets(thrust::cuda::dispatchable<System> &system,
                                        Size num_partitions,
                                        Size partition_size,
                                        InputIterator1 first1, InputIterator1 last1,
                                        InputIterator2 first2, InputIterator2 last2,
                                        OutputIterator result,
                                        Compare comp)
{
  find_partition_offsets_functor<Size,InputIterator1,InputIterator2,Compare> f(partition_size, first1, last1, first2, last2, comp);

  return thrust::transform(system,
                           thrust::counting_iterator<Size>(0),
                           thrust::counting_iterator<Size>(num_partitions),
                           result,
                           f);
}


namespace block
{


template<unsigned int block_size, typename T>
inline __device__
T right_neighbor(statically_blocked_thread_array<block_size> &ctx, const T &x, const T &boundary)
{
  // stage this shift to conserve smem
  const unsigned int storage_size = block_size / 2;
  __shared__ uninitialized_array<T,storage_size> shared;

  T result = x;

  unsigned int tid = ctx.thread_index();

  if(0 < tid && tid <= storage_size)
  {
    shared[tid - 1] = x;
  }

  ctx.barrier();

  if(tid < storage_size)
  {
    result = shared[tid];
  }

  ctx.barrier();
  
  tid -= storage_size;
  if(0 < tid && tid <= storage_size)
  {
    shared[tid - 1] = x;
  }
  else if(tid == 0)
  {
    shared[storage_size-1] = boundary;
  }

  ctx.barrier();

  if(tid < storage_size)
  {
    result = shared[tid];
  }

  ctx.barrier();

  return result;
}


template<uint16_t block_size, uint16_t work_per_thread, typename InputIterator1, typename InputIterator2, typename Compare, typename SetOperation>
inline __device__
  unsigned int bounded_count_set_operation_n(statically_blocked_thread_array<block_size> &ctx,
                                             InputIterator1 first1, uint16_t n1,
                                             InputIterator2 first2, uint16_t n2,
                                             Compare comp,
                                             SetOperation set_op)
{
  unsigned int thread_idx = ctx.thread_index();

  // find partition offsets
  uint16_t diag = thrust::min<uint16_t>(n1 + n2, thread_idx * work_per_thread);
  thrust::pair<uint16_t,uint16_t> thread_input_begin = balanced_path(first1, n1, first2, n2, diag, 2, comp);
  thrust::pair<uint16_t,uint16_t> thread_input_end   = block::right_neighbor<block_size>(ctx, thread_input_begin, thrust::make_pair(n1,n2));

  __shared__ uint16_t s_thread_output_size[block_size];

  // work_per_thread + 1 to accomodate a "starred" partition returned from balanced_path above
  s_thread_output_size[thread_idx] =
    set_op.count(work_per_thread + 1,
                 first1 + thread_input_begin.first,  first1 + thread_input_end.first,
                 first2 + thread_input_begin.second, first2 + thread_input_end.second,
                 comp);

  ctx.barrier();

  // reduce per-thread counts
  thrust::system::cuda::detail::block::inplace_inclusive_scan(ctx, s_thread_output_size);
  return s_thread_output_size[ctx.block_dimension() - 1];
}


inline __device__ int pop_count(unsigned int x)
{
// guard use of __popc from other compilers
#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC
  return __popc(x);
#else
  return x;
#endif
}



template<uint16_t block_size, uint16_t work_per_thread, typename InputIterator1, typename InputIterator2, typename OutputIterator, typename Compare, typename SetOperation>
inline __device__
  OutputIterator bounded_set_operation_n(statically_blocked_thread_array<block_size> &ctx,
                                         InputIterator1 first1, uint16_t n1,
                                         InputIterator2 first2, uint16_t n2,
                                         OutputIterator result,
                                         Compare comp,
                                         SetOperation set_op)
{
  unsigned int thread_idx = ctx.thread_index();
  
  // find partition offsets
  uint16_t diag = thrust::min<uint16_t>(n1 + n2, thread_idx * work_per_thread);
  thrust::pair<uint16_t,uint16_t> thread_input_begin = balanced_path(first1, n1, first2, n2, diag, 2, comp);
  thrust::pair<uint16_t,uint16_t> thread_input_end   = block::right_neighbor<block_size>(ctx, thread_input_begin, thrust::make_pair(n1,n2));

  typedef typename thrust::iterator_value<InputIterator1>::type value_type;
  // +1 to accomodate a "starred" partition returned from balanced_path above
  uninitialized_array<value_type, work_per_thread + 1> sparse_result;
  uint32_t active_mask =
    set_op(work_per_thread + 1,
           first1 + thread_input_begin.first,  first1 + thread_input_end.first,
           first2 + thread_input_begin.second, first2 + thread_input_end.second,
           sparse_result.begin(),
           comp);

  __shared__ uint16_t s_thread_output_size[block_size];
  s_thread_output_size[thread_idx] = pop_count(active_mask);

  ctx.barrier();

  // scan to turn per-thread counts into output indices
  uint16_t block_output_size = thrust::system::cuda::detail::block::inplace_exclusive_scan(ctx, s_thread_output_size, 0u);

  serial_bounded_copy_if(work_per_thread + 1, sparse_result.begin(), active_mask, result + s_thread_output_size[thread_idx]);

  ctx.barrier();

  return result + block_output_size;
}


template<uint16_t block_size, uint16_t work_per_thread, typename InputIterator1, typename InputIterator2, typename Compare, typename SetOperation>
inline __device__
  typename thrust::iterator_difference<InputIterator1>::type
    count_set_operation(statically_blocked_thread_array<block_size> &ctx,
                        InputIterator1 first1, InputIterator1 last1,
                        InputIterator2 first2, InputIterator2 last2,
                        Compare comp,
                        SetOperation set_op)
{
  typedef typename thrust::iterator_difference<InputIterator1>::type difference;

  difference result = 0;

  thrust::pair<difference,difference> remaining_input_size = thrust::make_pair(last1 - first1, last2 - first2);

  // iterate until the input is consumed
  while(remaining_input_size.first + remaining_input_size.second > 0)
  {
    // find the end of this subpartition's input
    // -1 to accomodate "starred" partitions
    uint16_t max_subpartition_size = block_size * work_per_thread - 1;
    difference diag = thrust::min<difference>(remaining_input_size.first + remaining_input_size.second, max_subpartition_size);
    thrust::pair<uint16_t,uint16_t> subpartition_size = balanced_path(first1, remaining_input_size.first, first2, remaining_input_size.second, diag, 4ll, comp);
  
    typedef typename thrust::iterator_value<InputIterator2>::type value_type;
    if(stage_through_smem<value_type>::value)
    {
      // load the input into __shared__ storage
      __shared__ uninitialized_array<value_type, block_size * work_per_thread> s_input;
  
      value_type *s_input_end1 = thrust::system::cuda::detail::block::copy_n(ctx, first1, subpartition_size.first,  s_input.begin());
      value_type *s_input_end2 = thrust::system::cuda::detail::block::copy_n(ctx, first2, subpartition_size.second, s_input_end1);
  
      result += block::bounded_count_set_operation_n<block_size,work_per_thread>(ctx,
                                                                                 s_input.begin(), subpartition_size.first,
                                                                                 s_input_end1,    subpartition_size.second,
                                                                                 comp,
                                                                                 set_op);
    }
    else
    {
      result += block::bounded_count_set_operation_n<block_size,work_per_thread>(ctx,
                                                                                 first1, subpartition_size.first,
                                                                                 first2, subpartition_size.second,
                                                                                 comp,
                                                                                 set_op);
    }

    // advance input
    first1 += subpartition_size.first;
    first2 += subpartition_size.second;

    // decrement remaining size
    remaining_input_size.first  -= subpartition_size.first;
    remaining_input_size.second -= subpartition_size.second;
  }

  return result;
}


template<uint16_t block_size, uint16_t work_per_thread, typename InputIterator1, typename InputIterator2, typename OutputIterator, typename Compare, typename SetOperation>
inline __device__
OutputIterator set_operation(statically_blocked_thread_array<block_size> &ctx,
                             InputIterator1 first1, InputIterator1 last1,
                             InputIterator2 first2, InputIterator2 last2,
                             OutputIterator result,
                             Compare comp,
                             SetOperation set_op)
{
  typedef typename thrust::iterator_difference<InputIterator1>::type difference;

  thrust::pair<difference,difference> remaining_input_size = thrust::make_pair(last1 - first1, last2 - first2);

  // iterate until the input is consumed
  while(remaining_input_size.first + remaining_input_size.second > 0)
  {
    // find the end of this subpartition's input
    // -1 to accomodate "starred" partitions
    uint16_t max_subpartition_size = block_size * work_per_thread - 1;
    difference diag = thrust::min<difference>(remaining_input_size.first + remaining_input_size.second, max_subpartition_size);
    thrust::pair<uint16_t,uint16_t> subpartition_size = balanced_path(first1, remaining_input_size.first, first2, remaining_input_size.second, diag, 4ll, comp);
    
    typedef typename thrust::iterator_value<InputIterator2>::type value_type;
    if(stage_through_smem<value_type>::value)
    {
      // load the input into __shared__ storage
      __shared__ uninitialized_array<value_type, block_size * work_per_thread> s_input;
  
      value_type *s_input_end1 = thrust::system::cuda::detail::block::copy_n(ctx, first1, subpartition_size.first,  s_input.begin());
      value_type *s_input_end2 = thrust::system::cuda::detail::block::copy_n(ctx, first2, subpartition_size.second, s_input_end1);
  
      result = block::bounded_set_operation_n<block_size,work_per_thread>(ctx,
                                                                          s_input.begin(), subpartition_size.first,
                                                                          s_input_end1,    subpartition_size.second,
                                                                          result,
                                                                          comp,
                                                                          set_op);
    }
    else
    {
      result = block::bounded_set_operation_n<block_size,work_per_thread>(ctx,
                                                                          first1, subpartition_size.first,
                                                                          first2, subpartition_size.second,
                                                                          result,
                                                                          comp,
                                                                          set_op);
    }
  
    // advance input
    first1 += subpartition_size.first;
    first2 += subpartition_size.second;

    // decrement remaining size
    remaining_input_size.first  -= subpartition_size.first;
    remaining_input_size.second -= subpartition_size.second;
  }

  return result;
}


} // end namespace block


template<uint16_t threads_per_block, uint16_t work_per_thread, typename InputIterator1, typename Size, typename InputIterator2, typename InputIterator3, typename OutputIterator, typename Compare, typename SetOperation>
  inline __device__ void count_set_operation(statically_blocked_thread_array<threads_per_block> &ctx,
                                             InputIterator1                                      input_partition_offsets,
                                             Size                                                num_partitions,
                                             InputIterator2                                      first1,
                                             InputIterator3                                      first2,
                                             OutputIterator                                      result,
                                             Compare                                             comp,
                                             SetOperation                                        set_op)
{
  // consume partitions
  for(Size partition_idx = ctx.block_index();
      partition_idx < num_partitions;
      partition_idx += ctx.grid_dimension())
  {
    typedef typename thrust::iterator_difference<InputIterator2>::type difference;

    // find the partition
    thrust::pair<difference,difference> block_input_begin = input_partition_offsets[partition_idx];
    thrust::pair<difference,difference> block_input_end   = input_partition_offsets[partition_idx + 1];

    // count the size of the set operation
    difference count = block::count_set_operation<threads_per_block,work_per_thread>(ctx,
                                                                                     first1 + block_input_begin.first,  first1 + block_input_end.first,
                                                                                     first2 + block_input_begin.second, first2 + block_input_end.second,
                                                                                     comp,
                                                                                     set_op);

    if(ctx.thread_index() == 0)
    {
      result[partition_idx] = count;
    }
  }
}


template<uint16_t threads_per_block, uint16_t work_per_thread, typename InputIterator1, typename Size, typename InputIterator2, typename InputIterator3, typename OutputIterator, typename Compare, typename SetOperation>
  struct count_set_operation_closure
{
  typedef statically_blocked_thread_array<threads_per_block> context_type;

  InputIterator1 input_partition_offsets;
  Size           num_partitions;
  InputIterator2 first1;
  InputIterator3 first2;
  OutputIterator result;
  Compare        comp;
  SetOperation   set_op;

  count_set_operation_closure(InputIterator1 input_partition_offsets,
                              Size           num_partitions,
                              InputIterator2 first1,
                              InputIterator3 first2,
                              OutputIterator result,
                              Compare        comp,
                              SetOperation   set_op)
    : input_partition_offsets(input_partition_offsets),
      num_partitions(num_partitions),
      first1(first1),
      first2(first2),
      result(result),
      comp(comp),
      set_op(set_op)
  {}

  inline __device__ void operator()() const
  {
    context_type ctx;
    count_set_operation<threads_per_block,work_per_thread>(ctx, input_partition_offsets, num_partitions, first1, first2, result, comp, set_op);
  }
};


template<uint16_t threads_per_block, uint16_t work_per_thread, typename InputIterator1, typename Size, typename InputIterator2, typename InputIterator3, typename OutputIterator, typename Compare, typename SetOperation>
  count_set_operation_closure<threads_per_block,work_per_thread,InputIterator1,Size,InputIterator2,InputIterator3,OutputIterator,Compare,SetOperation>
    make_count_set_operation_closure(InputIterator1 input_partition_offsets,
                                     Size           num_partitions,
                                     InputIterator2 first1,
                                     InputIterator3 first2,
                                     OutputIterator result,
                                     Compare        comp,
                                     SetOperation   set_op)
{
  typedef count_set_operation_closure<threads_per_block,work_per_thread,InputIterator1,Size,InputIterator2,InputIterator3,OutputIterator,Compare,SetOperation> result_type;
  return result_type(input_partition_offsets,num_partitions,first1,first2,result,comp,set_op);
}


template<uint16_t threads_per_block, uint16_t work_per_thread, typename InputIterator1, typename Size, typename InputIterator2, typename InputIterator3, typename InputIterator4, typename OutputIterator, typename Compare, typename SetOperation>
inline __device__
  void set_operation(statically_blocked_thread_array<threads_per_block> &ctx,
                     InputIterator1                                      input_partition_offsets,
                     Size                                                num_partitions,
                     InputIterator2                                      first1,
                     InputIterator3                                      first2,
                     InputIterator4                                      output_partition_offsets,
                     OutputIterator                                      result,
                     Compare                                             comp,
                     SetOperation                                        set_op)
{
  // consume partitions
  for(Size partition_idx = ctx.block_index();
      partition_idx < num_partitions;
      partition_idx += ctx.grid_dimension())
  {
    typedef typename thrust::iterator_difference<InputIterator2>::type difference;

    // find the partition
    thrust::pair<difference,difference> block_input_begin = input_partition_offsets[partition_idx];
    thrust::pair<difference,difference> block_input_end   = input_partition_offsets[partition_idx + 1];

    // do the set operation across the partition
    block::set_operation<threads_per_block,work_per_thread>(ctx,
                                                            first1 + block_input_begin.first,  first1 + block_input_end.first,
                                                            first2 + block_input_begin.second, first2 + block_input_end.second,
                                                            result + output_partition_offsets[partition_idx],
                                                            comp,
                                                            set_op);
  }
}


template<uint16_t threads_per_block, uint16_t work_per_thread, typename InputIterator1, typename Size, typename InputIterator2, typename InputIterator3, typename InputIterator4, typename OutputIterator, typename Compare, typename SetOperation>
  struct set_operation_closure
{
  typedef statically_blocked_thread_array<threads_per_block> context_type;

  InputIterator1 input_partition_offsets;
  Size           num_partitions;
  InputIterator2 first1;
  InputIterator3 first2;
  InputIterator4 output_partition_offsets;
  OutputIterator result;
  Compare        comp;
  SetOperation   set_op;

  set_operation_closure(InputIterator1 input_partition_offsets,
                        Size           num_partitions,
                        InputIterator2 first1,
                        InputIterator3 first2,
                        InputIterator4 output_partition_offsets,
                        OutputIterator result,
                        Compare        comp,
                        SetOperation   set_op)
    : input_partition_offsets(input_partition_offsets),
      num_partitions(num_partitions),
      first1(first1),
      first2(first2),
      output_partition_offsets(output_partition_offsets),
      result(result),
      comp(comp),
      set_op(set_op)
  {}

  inline __device__ void operator()() const
  {
    context_type ctx;
    set_operation<threads_per_block,work_per_thread>(ctx, input_partition_offsets, num_partitions, first1, first2, output_partition_offsets, result, comp, set_op);
  }
};


template<uint16_t threads_per_block, uint16_t work_per_thread, typename InputIterator1, typename Size, typename InputIterator2, typename InputIterator3, typename InputIterator4, typename OutputIterator, typename Compare, typename SetOperation>
  set_operation_closure<threads_per_block,work_per_thread,InputIterator1,Size,InputIterator2,InputIterator3,InputIterator4,OutputIterator,Compare,SetOperation>
    make_set_operation_closure(InputIterator1 input_partition_offsets,
                               Size           num_partitions,
                               InputIterator2 first1,
                               InputIterator3 first2,
                               InputIterator4 output_partition_offsets,
                               OutputIterator result,
                               Compare        comp,
                               SetOperation   set_op)
{
  typedef set_operation_closure<threads_per_block,work_per_thread,InputIterator1,Size,InputIterator2,InputIterator3,InputIterator4,OutputIterator,Compare,SetOperation> result_type;
  return result_type(input_partition_offsets,num_partitions,first1,first2,output_partition_offsets,result,comp,set_op);
}


} // end namespace set_operation_detail


template<typename System, typename InputIterator1, typename InputIterator2, typename OutputIterator, typename Compare, typename SetOperation>
  OutputIterator set_operation(thrust::cuda::dispatchable<System> &system,
                               InputIterator1 first1, InputIterator1 last1,
                               InputIterator2 first2, InputIterator2 last2,
                               OutputIterator result,
                               Compare comp,
                               SetOperation set_op)
{
  using thrust::system::cuda::detail::device_properties;
  using thrust::system::cuda::detail::detail::launch_closure;
  namespace d = thrust::system::cuda::detail::detail::set_operation_detail;

  typedef typename thrust::iterator_difference<InputIterator1>::type difference;

  const difference n1 = last1 - first1;
  const difference n2 = last2 - first2;

  // handle empty input
  if(n1 == 0 && n2 == 0)
  {
    return result;
  }

  const thrust::detail::uint16_t work_per_thread   = 15;
  const thrust::detail::uint16_t threads_per_block = 128;
  const thrust::detail::uint16_t work_per_block    = threads_per_block * work_per_thread;

  // -1 because balanced_path adds a single element to the end of a "starred" partition, increasing its size by one
  const thrust::detail::uint16_t maximum_partition_size = work_per_block - 1;
  const difference num_partitions = thrust::detail::util::divide_ri(n1 + n2, maximum_partition_size);

  // find input partition offsets
  // +1 to handle the end of the input elegantly
  thrust::detail::temporary_array<thrust::pair<difference,difference>, System> input_partition_offsets(0, system, num_partitions + 1);
  d::find_partition_offsets<difference>(system, input_partition_offsets.size(), maximum_partition_size, first1, last1, first2, last2, input_partition_offsets.begin(), comp);

  const difference num_blocks = thrust::min<difference>(device_properties().maxGridSize[0], num_partitions);

  // find output partition offsets
  // +1 to store the total size of the total
  thrust::detail::temporary_array<difference, System> output_partition_offsets(0, system, num_partitions + 1);
  launch_closure(d::make_count_set_operation_closure<threads_per_block,work_per_thread>(input_partition_offsets.begin(), num_partitions, first1, first2, output_partition_offsets.begin(), comp, set_op),
                 num_blocks,
                 threads_per_block);

  // turn the output partition counts into offsets to output partitions
  thrust::exclusive_scan(system, output_partition_offsets.begin(), output_partition_offsets.end(), output_partition_offsets.begin());

  // run the set op kernel
  launch_closure(d::make_set_operation_closure<threads_per_block,work_per_thread>(input_partition_offsets.begin(), num_partitions, first1, first2, output_partition_offsets.begin(), result, comp, set_op),
                 num_blocks,
                 threads_per_block);

  return result + output_partition_offsets[num_partitions];
}


} // end namespace detail
} // end namespace detail
} // end namespace cuda
} // end namespace system
} // end namespace thrust

