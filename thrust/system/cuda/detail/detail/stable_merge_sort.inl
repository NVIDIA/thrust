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
#include <thrust/system/cuda/detail/detail/stable_merge_sort.h>
#include <thrust/system/cuda/detail/detail/stable_sort_each.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/detail/temporary_array.h>
#include <thrust/system/cuda/detail/detail/launch_closure.h>
#include <thrust/system/cuda/detail/detail/virtualized_smem_closure.h>
#include <thrust/system/cuda/detail/merge.h>
#include <thrust/system/cuda/detail/extern_shared_ptr.h>
#include <thrust/detail/copy.h>
#include <thrust/tabulate.h>
#include <thrust/tuple.h>
#include <thrust/detail/minmax.h>
#include <thrust/detail/function.h>
#include <thrust/detail/integer_math.h>
#include <thrust/detail/integer_traits.h>
#include <thrust/detail/seq.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/detail/internal_functional.h>
#include <thrust/system/cuda/detail/temporary_indirect_permutation.h>
#include <thrust/system/cuda/detail/runtime_introspection.h>


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
namespace stable_merge_sort_detail
{
namespace block
{


// block-wise inplace merge for when we have a static bound on the size of the result (block_size * work_per_thread)
template<unsigned int work_per_thread,
         typename Context,
         typename Iterator,
         typename Size,
         typename Compare>
__device__
void bounded_inplace_merge(Context &ctx, Iterator first, Size n1, Size n2, Compare comp)
{
  Iterator first2 = first + n1;

  // don't ask for an out-of-bounds diagonal
  Size diag = thrust::min<Size>(n1 + n2, work_per_thread * ctx.thread_index());

  Size mp = merge_path(diag, first, n1, first2, n2, comp);

  // compute the ranges of the sources
  Size start1 = mp;
  Size start2 = diag - mp;

  Size end1 = n1;
  Size end2 = n2;
  
  // each thread does a local sequential merge
  typedef typename thrust::iterator_value<Iterator>::type value_type;
  value_type local_result[work_per_thread];
  sequential_bounded_merge<work_per_thread>(first  + start1, first  + end1,
                                            first2 + start2, first2 + end2,
                                            local_result, comp);

  ctx.barrier();

  // store the result
  // XXX we unconditionally copy work_per_thread elements here, even if input was partially-sized
  thrust::copy_n(thrust::seq, local_result, work_per_thread, first + work_per_thread * ctx.thread_index());
  ctx.barrier();
}


// staged, block-wise merge for when we have a static bound on the size of the result (block_size * work_per_thread)
template<unsigned int work_per_thread,
         typename Context,
         typename Iterator1, typename Size1,
         typename Iterator2, typename Size2,
         typename Iterator3,
         typename Iterator4,
	 typename Compare>
__device__
void staged_bounded_merge(Context &ctx,
                          Iterator1 first1, Size1 n1,
                          Iterator2 first2, Size2 n2,
                          Iterator3 staging_buffer,
                          Iterator4 result,
                          Compare comp)
{
  // stage the input through the buffer
  cuda::detail::block::async_copy_n_global_to_shared<work_per_thread>(ctx, first1, n1, staging_buffer);
  cuda::detail::block::async_copy_n_global_to_shared<work_per_thread>(ctx, first2, n2, staging_buffer + n1);
  ctx.barrier();

  // cooperatively merge in place
  block::bounded_inplace_merge<work_per_thread>(ctx, staging_buffer, n1, n2, comp);
  
  // store result in buffer to result
  cuda::detail::block::copy_n(ctx, staging_buffer, n1 + n2, result);
}


} // end block


// Returns (start1, end1, start2, end2) into mergesort input lists between mp0 and mp1.
inline __host__ __device__
thrust::tuple<int,int,int,int> find_mergesort_interval(int partition_first1, int partition_size, int num_blocks_per_merge, int block_idx, int num_elements_per_block, int n, int mp, int right_mp)
{
  int partition_first2 = partition_first1 + partition_size;

  // Locate diag from the start of the A sublist.
  int diag = num_elements_per_block * block_idx - partition_first1;
  int start1 = partition_first1 + mp;
  int end1 = thrust::min<int>(n, partition_first1 + right_mp);
  int start2 = thrust::min<int>(n, partition_first2 + diag - mp);
  int end2 = thrust::min<int>(n, partition_first2 + diag + num_elements_per_block - right_mp);
  
  // The end partition of the last block for each merge operation is computed
  // and stored as the begin partition for the subsequent merge. i.e. it is
  // the same partition but in the wrong coordinate system, so its 0 when it
  // should be listSize. Correct that by checking if this is the last block
  // in this merge operation.
  if(num_blocks_per_merge - 1 == ((num_blocks_per_merge - 1) & block_idx))
  {
    end1 = thrust::min<int>(n, partition_first1 + partition_size);
    end2 = thrust::min<int>(n, partition_first2 + partition_size);
  }

  return thrust::make_tuple(start1, end1, start2, end2);
}


inline __host__ __device__
thrust::tuple<int,int,int,int> locate_merge_partitions(int n, int block_idx, int num_blocks_per_merge, int num_elements_per_block, int mp, int right_mp)
{
  int first_block_in_partition = ~(num_blocks_per_merge - 1) & block_idx;
  int partition_size = num_elements_per_block * (num_blocks_per_merge >> 1);

  int partition_first1 = num_elements_per_block * first_block_in_partition;

  return find_mergesort_interval(partition_first1, partition_size, num_blocks_per_merge, block_idx, num_elements_per_block, n, mp, right_mp);
}


template<unsigned int work_per_thread,
         typename Context,
         typename Size,
         typename Iterator1,
         typename Iterator2,
         typename Iterator3,
         typename Compare>
struct merge_adjacent_partitions_closure
{
  typedef Context context_type;

  Size num_blocks_per_merge;
  Iterator1 first;
  Size n;
  Iterator2 merge_paths;
  Iterator3 result;
  thrust::detail::wrapped_function<Compare,bool> comp;


  __host__ __device__
  merge_adjacent_partitions_closure(Size num_blocks_per_merge, Iterator1 first, Size n, Iterator2 merge_paths, Iterator3 result, Compare comp)
    : num_blocks_per_merge(num_blocks_per_merge),
      first(first),
      n(n),
      merge_paths(merge_paths),
      result(result),
      comp(comp)
  {}


  template<typename RandomAccessIterator>
  __thrust_forceinline__ __device__
  void operator()(RandomAccessIterator staging_buffer)
  {
    context_type ctx;

    Size work_per_block = ctx.block_dimension() * work_per_thread;
    
    Size start1 = 0, end1 = 0, start2 = 0, end2 = 0;

    thrust::tie(start1,end1,start2,end2) =
      locate_merge_partitions(n, ctx.block_index(), num_blocks_per_merge, work_per_block, merge_paths[ctx.block_index()], merge_paths[ctx.block_index() + 1]);

    block::staged_bounded_merge<work_per_thread>(ctx,
                                                 first + start1, end1 - start1,
                                                 first + start2, end2 - start2,
                                                 staging_buffer,
                                                 result + ctx.block_index() * work_per_block,
                                                 comp);
  }


  __thrust_forceinline__ __device__
  void operator()()
  {
    typedef typename thrust::iterator_value<Iterator1>::type value_type;

    // stage this operation through smem
    // the size of this array is block_size * (work_per_thread + 1)
    value_type *s_keys = thrust::system::cuda::detail::extern_shared_ptr<value_type>();
    
    this->operator()(s_keys);
  }
};


template<unsigned int work_per_thread,
         typename DerivedPolicy,
         typename Context,
         typename Size,
         typename Iterator1,
         typename Iterator2,
         typename Pointer,
         typename Iterator3,
         typename Compare>
__host__ __device__
void merge_adjacent_partitions(thrust::system::cuda::execution_policy<DerivedPolicy> &exec,
                               Context context,
                               unsigned int block_size,
                               Size num_blocks_per_merge,
                               Iterator1 first,
                               Size n,
                               Iterator2 merge_paths,
                               Pointer virtual_smem,
                               Iterator3 result,
                               Compare comp)
{
  typedef merge_adjacent_partitions_closure<
    work_per_thread,
    Context,
    Size,
    Iterator1,
    Iterator2,
    Iterator3,
    Compare
  > closure_type;

  closure_type closure(num_blocks_per_merge, first, n, merge_paths, result, comp);

  Size num_blocks = thrust::detail::util::divide_ri(n, block_size * work_per_thread);

  typedef typename thrust::iterator_value<Iterator1>::type value_type;

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


template<typename Iterator, typename Size, typename Compare>
struct locate_merge_path
{
  Iterator haystack_first;
  Size haystack_size;
  Size num_elements_per_block;
  Size num_blocks_per_merge;
  thrust::detail::wrapped_function<Compare,bool> comp;

  __host__ __device__
  locate_merge_path(Iterator haystack_first, Size haystack_size, Size num_elements_per_block, Size num_blocks_per_merge, Compare comp)
    : haystack_first(haystack_first),
      haystack_size(haystack_size),
      num_elements_per_block(num_elements_per_block),
      num_blocks_per_merge(num_blocks_per_merge),
      comp(comp)
  {}

  template<typename Index>
  __host__ __device__
  Index operator()(Index merge_path_idx)
  {
    // find the index of the first CTA that will participate in the eventual merge
    Size first_block_in_partition = ~(num_blocks_per_merge - 1) & merge_path_idx;

    // the size of each block's input
    Size size = num_elements_per_block * (num_blocks_per_merge / 2);

    // find pointers to the two input arrays
    Size start1 = num_elements_per_block * first_block_in_partition;
    Size start2 = thrust::min<Size>(haystack_size, start1 + size);

    // the size of each input array
    // note we clamp to the end of the total input to handle the last partial list
    Size n1 = thrust::min<Size>(size, haystack_size - start1);
    Size n2 = thrust::min<Size>(size, haystack_size - start2);
    
    // note that diag is computed as an offset from the beginning of the first list
    Size diag = thrust::min<Size>(n1 + n2, num_elements_per_block * merge_path_idx - start1);

    return merge_path(diag, haystack_first + start1, n1, haystack_first + start2, n2, comp);
  }
};


template<typename DerivedPolicy, typename Iterator1, typename Size1, typename Iterator2, typename Size2, typename Compare>
__host__ __device__
void locate_merge_paths(thrust::system::cuda::execution_policy<DerivedPolicy> &exec,
                        Iterator1 result,
                        Size1 n,
                        Iterator2 haystack_first,
                        Size2 haystack_size,
                        Size2 num_elements_per_block,
                        Size2 num_blocks_per_merge,
                        Compare comp)
{
  locate_merge_path<Iterator2,Size2,Compare> f(haystack_first, haystack_size, num_elements_per_block, num_blocks_per_merge, comp);

  thrust::tabulate(exec, result, result + n, f);
}


template<typename T>
__host__ __device__
bool virtualize_smem(size_t num_elements_per_block)
{
#ifndef __CUDA_ARCH__
  size_t num_smem_bytes_required = num_elements_per_block * sizeof(T);

  thrust::system::cuda::detail::device_properties_t props = thrust::system::cuda::detail::device_properties();

  size_t num_smem_bytes_available = props.sharedMemPerBlock;
  if(props.major == 1)
  {
    // pay the kernel parameters tax on Tesla
    num_smem_bytes_available -= 256;
  }

  return num_smem_bytes_required > num_smem_bytes_available;
#else
  // we should never need to virtualize smem on anything besides Tesla,
  // and Tesla will never execute this code path
  return false;
#endif
}


template<typename DerivedPolicy, typename RandomAccessIterator, typename Size, typename Compare>
__host__ __device__
void stable_merge_sort_n(thrust::system::cuda::execution_policy<DerivedPolicy> &exec,
                         RandomAccessIterator first,
                         Size n,
                         Compare comp)
{
  if(n <= 0) return;

  typedef typename thrust::iterator_value<RandomAccessIterator>::type T;

  const Size block_size = 256;

  typedef thrust::system::cuda::detail::detail::statically_blocked_thread_array<block_size> context_type;

  context_type context;

  const Size work_per_thread = (sizeof(T) < 8) ?  11 : 7;
  const Size work_per_block = block_size * work_per_thread;

  Size num_blocks = thrust::detail::util::divide_ri(n, work_per_block);

  const unsigned int num_smem_elements_per_block = block_size * (work_per_thread + 1);

  thrust::detail::temporary_array<T,DerivedPolicy> virtual_smem(exec, virtualize_smem<T>(num_smem_elements_per_block) ? (num_blocks * num_smem_elements_per_block) : 0);
  
  // depending on the number of passes
  // we'll either do the initial segmented sort inplace or not
  // ping being true means the latest data is in the source array
  bool ping = false;
  thrust::detail::temporary_array<T,DerivedPolicy> pong_buffer(exec, n);

  Size num_passes = thrust::detail::log2_ri(num_blocks);

  if(thrust::detail::is_odd(num_passes))
  {
    stable_sort_each_copy<work_per_thread>(exec, context, block_size, first, first + n, thrust::raw_pointer_cast(&*virtual_smem.begin()), pong_buffer.begin(), comp);
  }
  else
  {
    stable_sort_each_copy<work_per_thread>(exec, context, block_size, first, first + n, thrust::raw_pointer_cast(&*virtual_smem.begin()), first, comp);
    ping = true;
  }

  thrust::detail::temporary_array<Size,DerivedPolicy> merge_paths(exec, num_blocks + 1);
  
  for(Size pass = 0; pass < num_passes; ++pass, ping = !ping)
  {
    Size num_blocks_per_merge = 2 << pass;

    if(ping)
    {
      locate_merge_paths(exec, merge_paths.begin(), merge_paths.size(), first, n, work_per_block, num_blocks_per_merge, comp);

      merge_adjacent_partitions<work_per_thread>(exec, context, block_size, num_blocks_per_merge, first, n, merge_paths.begin(), thrust::raw_pointer_cast(&*virtual_smem.begin()), pong_buffer.begin(), comp);
    }
    else
    {
      locate_merge_paths(exec, merge_paths.begin(), merge_paths.size(), pong_buffer.begin(), n, work_per_block, num_blocks_per_merge, comp);

      merge_adjacent_partitions<work_per_thread>(exec, context, block_size, num_blocks_per_merge, pong_buffer.begin(), n, merge_paths.begin(), thrust::raw_pointer_cast(&*virtual_smem.begin()), first, comp);
    }
  }
}


template<typename DerivedPolicy, typename RandomAccessIterator, typename Compare>
__host__ __device__
void stable_merge_sort(thrust::system::cuda::execution_policy<DerivedPolicy> &exec,
                       RandomAccessIterator first,
                       RandomAccessIterator last,
                       Compare comp)
{
  typedef typename thrust::iterator_difference<RandomAccessIterator>::type difference_type;

  difference_type n = last - first;

  // if difference_type is large and n can fit into a 32b uint then use that
  thrust::detail::uint32_t threshold = thrust::detail::integer_traits<thrust::detail::uint32_t>::const_max;
  if(sizeof(difference_type) > sizeof(thrust::detail::uint32_t) && n <= difference_type(threshold))
  {
    stable_merge_sort_n(exec, first, static_cast<thrust::detail::uint32_t>(n), comp);
  }
  else
  {
    stable_merge_sort_n(exec, first, n, comp);
  }
}


} // end namespace stable_merge_sort_detail


template<typename DerivedPolicy, typename RandomAccessIterator, typename Compare>
__host__ __device__
void stable_merge_sort(thrust::system::cuda::execution_policy<DerivedPolicy> &exec,
                       RandomAccessIterator first,
                       RandomAccessIterator last,
                       Compare comp)
{
  // decide whether to apply indirection
  typedef typename thrust::iterator_value<RandomAccessIterator>::type value_type;

  typedef thrust::detail::integral_constant<bool, (sizeof(value_type) > 16)> use_indirection;

  conditional_temporary_indirect_ordering<
    use_indirection,
    DerivedPolicy,
    RandomAccessIterator,
    Compare
  > potentially_indirect_range(exec, first, last, comp);

  stable_merge_sort_detail::stable_merge_sort(exec,
                                              potentially_indirect_range.begin(),
                                              potentially_indirect_range.end(),
                                              potentially_indirect_range.comp());
}


template<typename DerivedPolicy, typename RandomAccessIterator1, typename RandomAccessIterator2, typename Compare>
__host__ __device__
void stable_merge_sort_by_key(thrust::system::cuda::execution_policy<DerivedPolicy> &exec,
                              RandomAccessIterator1 keys_first,
                              RandomAccessIterator1 keys_last,
                              RandomAccessIterator2 values_first,
                              Compare comp)
{
  typedef thrust::tuple<RandomAccessIterator1,RandomAccessIterator2> iterator_tuple;
  typedef thrust::zip_iterator<iterator_tuple> zip_iterator;

  zip_iterator zipped_first = thrust::make_zip_iterator(thrust::make_tuple(keys_first, values_first));
  zip_iterator zipped_last = thrust::make_zip_iterator(thrust::make_tuple(keys_last, values_first));

  thrust::detail::compare_first<Compare> comp_first(comp);

  stable_merge_sort(exec, zipped_first, zipped_last, comp_first);
}


} // end namespace detail
} // end namespace detail
} // end namespace cuda
} // end namespace system
} // end namespace thrust

