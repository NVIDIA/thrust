/*
 *  Copyright 2008-2013 NVIDIA Corporation
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
#include <thrust/system/cuda/detail/detail/stable_sort_each.h>
#include <thrust/system/cuda/detail/block/copy.h>
#include <thrust/system/cuda/detail/detail/merge.h>
#include <thrust/system/cuda/detail/extern_shared_ptr.h>
#include <thrust/detail/minmax.h>
#include <thrust/detail/swap.h>
#include <thrust/detail/util/blocking.h>
#include <thrust/detail/integer_math.h>
#include <thrust/system/cuda/detail/detail/launch_closure.h>
#include <thrust/system/cuda/detail/execution_policy.h>
#include <thrust/system/cuda/detail/detail/virtualized_smem_closure.h>


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
namespace stable_sort_each_detail
{
namespace static_stable_odd_even_transpose_sort_detail
{


template<int i, int n>
struct impl
{
  template<typename Iterator, typename Compare>
  static __device__
  void do_it(Iterator keys, Compare comp)
  {
    for(int j = 1 & i; j < n - 1; j += 2)
    {
      if(comp(keys[j + 1], keys[j]))
      {
        using thrust::swap;

      	swap(keys[j], keys[j + 1]);
      }
    }

    impl<i + 1, n>::do_it(keys, comp);
  }
};


template<int i>
struct impl<i,i>
{
  template<typename Iterator, typename Compare>
  static __device__
  void do_it(Iterator, Compare) {}
};


} // end static_stable_odd_even_transpose_sort_detail


template<int n, typename RandomAccessIterator, typename Compare>
__device__
void static_stable_sort(RandomAccessIterator keys, Compare comp)
{
  static_stable_odd_even_transpose_sort_detail::impl<0,n>::do_it(keys, comp);
}


// sequential copy_n for when we have a static bound on the value of n
template<unsigned int bound_n, typename RandomAccessIterator1, typename Size, typename RandomAccessIterator2>
__device__
void bounded_copy_n(RandomAccessIterator1 first, Size n, RandomAccessIterator2 result)
{
  for(unsigned int i = 0; i < bound_n; ++i)
  {
    if(i < n)
    {
      result[i] = first[i];
    }
  }
}


namespace block
{


template<unsigned int work_per_thread, typename Context, typename Iterator, typename Size, typename Compare>
__device__
void bounded_inplace_merge_adjacent_partitions(Context &ctx,
                                               Iterator first,
                                               Size n,
                                               Compare comp)
{
  typedef typename thrust::iterator_value<Iterator>::type value_type;

  for(Size num_threads_per_merge = 2; num_threads_per_merge <= ctx.block_dimension(); num_threads_per_merge *= 2)
  {
    // find the index of the first array this thread will merge
    Size list = ~(num_threads_per_merge - 1) & ctx.thread_index();
    Size diag = thrust::min<Size>(n, work_per_thread * ((num_threads_per_merge - 1) & ctx.thread_index()));
    Size input_start = work_per_thread * list;

    // the size of each of the two input arrays we're merging
    Size input_size = work_per_thread * (num_threads_per_merge / 2);

    // find the limits of the partitions of the input this group of threads will merge
    Size partition_first1 = thrust::min<Size>(n, input_start);
    Size partition_first2 = thrust::min<Size>(n, partition_first1 + input_size); 
    Size partition_last2  = thrust::min<Size>(n, partition_first2 + input_size);

    Size n1 = partition_first2 - partition_first1;
    Size n2 = partition_last2  - partition_first2;

    Size mp = merge_path(diag, first + partition_first1, n1, first + partition_first2, n2, comp);

    // each thread merges sequentially locally
    value_type local_result[work_per_thread];
    sequential_bounded_merge<work_per_thread>(first + partition_first1 + mp,        first + partition_first2,
                                              first + partition_first2 + diag - mp, first + partition_last2,
                                              local_result,
                                              comp);

    ctx.barrier();

    // compute the size of the local result to account for the final, partial tile
    Size local_result_size = thrust::min<Size>(work_per_thread, n - (ctx.thread_index() * work_per_thread));

    // store local results
    bounded_copy_n<work_per_thread>(local_result, local_result_size, first + ctx.thread_index() * work_per_thread);

    ctx.barrier();
  }
}


template<unsigned int work_per_thread, typename Context, typename RandomAccessIterator, typename Size, typename Compare>
__device__
void bounded_stable_sort(Context &ctx,
                         RandomAccessIterator first,
                         Size n,
                         Compare comp)
{
  typedef typename thrust::iterator_value<RandomAccessIterator>::type value_type;

  // compute the size of this thread's local tile to account for the final, partial tile
  Size local_tile_size = work_per_thread;
  if(work_per_thread * (ctx.thread_index() + 1) > n)
  {
    local_tile_size = thrust::max<Size>(0, n - (work_per_thread * ctx.thread_index()));
  }

  // each thread creates a local copy of its partition of the array
  value_type local_keys[work_per_thread];
  bounded_copy_n<work_per_thread>(first + ctx.thread_index() * work_per_thread, local_tile_size, local_keys);
  
  // if we're in the final partial tile, fill the remainder of the local_keys with with the max value
  if(local_tile_size < work_per_thread)
  {
    value_type max_key = local_keys[0];

    for(unsigned int i = 1; i < work_per_thread; ++i)
    {
      if(i < local_tile_size)
      {
        max_key = comp(max_key, local_keys[i]) ? local_keys[i] : max_key;
      }
    }
    
    // fill in the remainder with max_key
    for(unsigned int i = 0; i < work_per_thread; ++i)
    {
      if(i >= local_tile_size)
      {
        local_keys[i] = max_key;
      }
    }
  }

  // stable sort the keys in the thread.
  if(work_per_thread * ctx.thread_index() < n)
  {
    static_stable_sort<work_per_thread>(local_keys, comp);
  }
  
  // Store the locally sorted keys into shared memory.
  bounded_copy_n<work_per_thread>(local_keys, local_tile_size, first + ctx.thread_index() * work_per_thread);
  ctx.barrier();

  block::bounded_inplace_merge_adjacent_partitions<work_per_thread>(ctx, first, n, comp);
}


} // end block


template<unsigned int work_per_thread,
         typename Context,
         typename RandomAccessIterator1,
         typename Size,
         typename RandomAccessIterator2,
         typename Compare>
struct stable_sort_each_copy_closure
{
  typedef Context context_type;

  RandomAccessIterator1 first;
  Size n;
  RandomAccessIterator2 result;
  thrust::detail::wrapped_function<Compare,bool> comp;

  __host__ __device__
  stable_sort_each_copy_closure(RandomAccessIterator1 first, Size n, RandomAccessIterator2 result, Compare comp)
    : first(first),
      n(n),
      result(result),
      comp(comp)
  {}


  template<typename RandomAccessIterator>
  __device__ __thrust_forceinline__
  void operator()(RandomAccessIterator staging_buffer)
  {
    context_type ctx;

    unsigned int work_per_block = ctx.block_dimension() * work_per_thread;
    unsigned int offset = work_per_block * ctx.block_index();
    unsigned int tile_size = thrust::min<unsigned int>(work_per_block, n - offset);
    
    // load input tile into buffer
    thrust::system::cuda::detail::block::copy_n_global_to_shared<work_per_thread>(ctx, first + offset, tile_size, staging_buffer);

    // sort input in buffer
    block::bounded_stable_sort<work_per_thread>(ctx, staging_buffer, tile_size, comp);
    
    // store result to gmem
    thrust::system::cuda::detail::block::copy_n(ctx, staging_buffer, tile_size, result + offset);
  }


  __device__ __thrust_forceinline__
  void operator()()
  {
    typedef typename thrust::iterator_value<RandomAccessIterator1>::type value_type;

    // stage this operation through smem
    // the size of this array is block_size * (work_per_thread + 1)
    value_type *s_keys = thrust::system::cuda::detail::extern_shared_ptr<value_type>();
    
    this->operator()(s_keys);
  }
};


} // end namespace stable_sort_each_detail


template<unsigned int work_per_thread,
         typename DerivedPolicy,
         typename Context,
         typename RandomAccessIterator1,
         typename Pointer,
         typename RandomAccessIterator2,
         typename Compare>
__host__ __device__
void stable_sort_each_copy(execution_policy<DerivedPolicy> &exec,
                           Context context,
                           unsigned int block_size,
                           RandomAccessIterator1 first, RandomAccessIterator1 last,
                           Pointer virtual_smem,
                           RandomAccessIterator2 result,
                           Compare comp)
{
  typedef typename thrust::iterator_difference<RandomAccessIterator1>::type difference_type;

  difference_type n = last - first;

  int num_blocks = thrust::detail::util::divide_ri(n, block_size * work_per_thread);

  typedef stable_sort_each_detail::stable_sort_each_copy_closure<
    work_per_thread,
    Context,
    RandomAccessIterator1,
    difference_type,
    RandomAccessIterator2,
    Compare
  > closure_type;

  closure_type closure(first, n, result, comp);
  
  typedef typename thrust::iterator_value<RandomAccessIterator1>::type value_type;

  const size_t num_smem_elements_per_block = block_size * (work_per_thread + 1);

  // XXX this virtualizing code can probably be generalized and moved elsewhere
  if(virtual_smem)
  {
    virtualized_smem_closure<closure_type, Pointer> virtualized_closure(closure, num_smem_elements_per_block, virtual_smem);

    thrust::system::cuda::detail::detail::launch_closure(exec, virtualized_closure, num_blocks, block_size);
  }
  else
  {
    const size_t num_smem_bytes = num_smem_elements_per_block * sizeof(value_type);

    thrust::system::cuda::detail::detail::launch_closure(exec, closure, num_blocks, block_size, num_smem_bytes);
  }
}


} // end detail
} // end detail
} // end cuda
} // end system
} // end thrust

