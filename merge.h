#include <thrust/detail/config.h>
#include <thrust/pair.h>
#include <thrust/detail/minmax.h>
#include <thrust/detail/function.h>
#include <thrust/system/cuda/detail/detail/uninitialized.h>
#include <thrust/detail/util/blocking.h>


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


template<typename RandomAccessIterator1, typename Size, typename RandomAccessIterator2, typename RandomAccessIterator3, typename Compare>
__global__
void merge_n(RandomAccessIterator1 first1,
             Size n1,
             RandomAccessIterator2 first2,
             Size n2,
             RandomAccessIterator3 result,
             Compare comp_,
             unsigned int work_per_thread)
{
  const unsigned int block_size = blockDim.x;
  thrust::detail::device_function<Compare,bool> comp;
  typedef typename thrust::iterator_value<RandomAccessIterator1>::type value_type1;
  typedef typename thrust::iterator_value<RandomAccessIterator2>::type value_type2;

  Size result_size = n1 + n2;

  // this is just oversubscription_rate * block_size * work_per_thread
  // but it makes no sense to send oversubscription_rate as an extra parameter
  Size work_per_block = thrust::detail::util::divide_ri(result_size, gridDim.x);

  using thrust::system::cuda::detail::detail::uninitialized;
  __shared__ uninitialized<thrust::pair<Size,Size> > block_input_begin;

  if(threadIdx.x == 0)
  {
    block_input_begin = (blockIdx.x == 0) ?
      thrust::make_pair(Size(0),Size(0)) :
      partition_search(first1, first2,
                       Size(work_per_block * blockIdx.x),
                       Size(0), n1,
                       Size(0), n2,
                       comp);
  }

  __syncthreads();

  thrust::pair<Size,Size> input_offset = block_input_begin;

  Size result_block_offset = blockIdx.x*work_per_block;
  Size result_block_offset_last = result_block_offset + thrust::min<Size>(work_per_block, result_size - result_block_offset);
  Size work_per_iteration = block_size * work_per_thread;

  for(;
      result_block_offset < result_block_offset_last;
      result_block_offset += work_per_iteration,
      input_offset.first  += work_per_iteration,
      input_offset.second += work_per_iteration
     )
  {
    thrust::pair<Size,Size> thread_input_begin =
      partition_search(first1, first2,
                       Size(result_block_offset + threadIdx.x*work_per_thread),
                       block_input_begin.get().first, thrust::min<Size>(input_offset.first + work_per_iteration, n1),
                       block_input_begin.get().second, thrust::min<Size>(input_offset.second + work_per_iteration, n2),
                       comp);

    uninitialized<value_type1> x1;
    if(thread_input_begin.first < n1)
    {
      x1 = first1[thread_input_begin.first];
    }

    uninitialized<value_type2> x2;
    if(thread_input_begin.second < n2)
    {
      x2 = first2[thread_input_begin.second];
    }

    // XXX this is just a serial merge -- simplify this loop
    Size i = result_block_offset + threadIdx.x * work_per_thread;
    Size last_i = i + thrust::min<Size>(work_per_thread, result_size - thread_input_begin.first - thread_input_begin.second);
    for(;
        i < last_i;
        ++i)
    {
       bool output_x2 = false;
       if(thread_input_begin.first >= n1)
       {
         output_x2 = true;
       }
       else if(thread_input_begin.second >= n2)
       {
         output_x2 = false;
       }
       else
       {
         output_x2 = comp(x2, x1);
       }

       result[i] = output_x2 ? x2.get() : x1.get();

       if(output_x2)
       {
         ++thread_input_begin.second;
         x2 = first2[thread_input_begin.second];
       }
       else
       {
         ++thread_input_begin.first;
         x1 = first1[thread_input_begin.first];
       }
    } // end for

    if(threadIdx.x == block_size-1)
    {
      block_input_begin = thread_input_begin;
    }
    __syncthreads();
  } // end for
} // end merge_n


// returns (work_per_thread, threads_per_block, oversubscription_factor)
template<typename RandomAccessIterator1, typename RandomAccessIterator2, typename RandomAccessIterator3, typename Compare>
  thrust::tuple<unsigned int,unsigned int,unsigned int>
    merge_tunables(RandomAccessIterator1, RandomAccessIterator1, RandomAccessIterator2, RandomAccessIterator2, RandomAccessIterator3, Compare comp)
{
  // determined by empirical testing on GTX 480
  // ~4500 Mkeys/s on GTX 480
  const unsigned int work_per_thread         = 5;
  const unsigned int threads_per_block       = 128;
  const unsigned int oversubscription_factor = 30;

  return thrust::make_tuple(work_per_thread, threads_per_block, oversubscription_factor);
}


template<typename RandomAccessIterator1, typename RandomAccessIterator2, typename RandomAccessIterator3, typename Compare>
  RandomAccessIterator3 new_merge(RandomAccessIterator1 first1, RandomAccessIterator1 last1, RandomAccessIterator2 first2, RandomAccessIterator2 last2, RandomAccessIterator3 result, Compare comp)
{
  typename thrust::iterator_difference<RandomAccessIterator1>::type n1 = last1 - first1;
  typename thrust::iterator_difference<RandomAccessIterator2>::type n2 = last2 - first2;

  unsigned int work_per_thread = 0, threads_per_block = 0, oversubscription_factor = 0;
  thrust::tie(work_per_thread,threads_per_block,oversubscription_factor)
    = merge_tunables(first1, last1, first2, last2, result, comp);

  const unsigned int work_per_block = work_per_thread * threads_per_block;

  typename thrust::iterator_difference<RandomAccessIterator1>::type n = n1 + n2;

  using thrust::system::cuda::detail::device_properties;
  const unsigned int num_processors = device_properties().multiProcessorCount;
  const unsigned int num_blocks = thrust::min<int>(oversubscription_factor * num_processors, thrust::detail::util::divide_ri(n, work_per_block));

  if(num_blocks > 0) merge_n<<<num_blocks, threads_per_block>>>(first1, n1, first2, n2, result, comp, work_per_thread);

  return result + n1 + n2;
}


