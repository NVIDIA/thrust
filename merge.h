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

  // XXX isn't this just block_size * work_per_thread ?
  Size per_cta_size = thrust::detail::util::divide_ri(result_size, gridDim.x);

  Size cta_portion = thrust::min<Size>(per_cta_size, result_size-blockIdx.x*per_cta_size);
  Size cta_offset = blockIdx.x*per_cta_size;

  using thrust::system::cuda::detail::detail::uninitialized;
  __shared__ uninitialized<thrust::pair<Size,Size> > block_begin;

  if(threadIdx.x == 0)
  {
    block_begin = (blockIdx.x == 0) ?
      thrust::make_pair(Size(0),Size(0)) :
      partition_search(first1, first2,
                       Size(per_cta_size * blockIdx.x),
                       Size(0), n1,
                       Size(0), n2,
                       comp);
  }

  __syncthreads();

  Size remaining_size = cta_portion;
  Size processed_size = 0;
  thrust::pair<Size,Size> offset = block_begin;

  while(remaining_size > 0)
  {
    thrust::pair<Size,Size> thread_begin =
      partition_search(first1, first2,
                       Size(cta_offset+processed_size+threadIdx.x*work_per_thread),
                       block_begin.get().first, thrust::min<Size>(offset.first+block_size*work_per_thread, n1),
                       block_begin.get().second, thrust::min<Size>(offset.second+block_size*work_per_thread, n2),
                       comp);

    //value_type1 x1 = __ldg(first1 + thread_begin.first);
    //value_type2 x2 = __ldg(first2 + thread_begin.second);

    uninitialized<value_type1> x1;
    if(thread_begin.first < n1)
    {
      x1 = first1[thread_begin.first];
    }

    uninitialized<value_type2> x2;
    if(thread_begin.second < n2)
    {
      x2 = first2[thread_begin.second];
    }

    Size count = thrust::min<Size>(work_per_thread, result_size - thread_begin.first - thread_begin.second);

    // XXX this is just a serial merge -- simplify this loop
    for(Size i = 0; i < count; i++)
    {
       bool p = false;
       if(thread_begin.second >= n2)
       {
         p = true;
       }
       else if(thread_begin.first >= n1)
       {
         p = false;
       }
       else
       {
         p = !comp(x2, x1);
       }

       // XXX lot of redundant arithmetic in this index -- why can't we simply use the loop counter?
       result[cta_offset+processed_size+threadIdx.x*work_per_thread+i] = p ? x1.get() : x2.get();

       if(p)
       {
         ++thread_begin.first;
         x1 = first1[thread_begin.first];
       }
       else
       {
         ++thread_begin.second;
         x2 = first2[thread_begin.second];
       }
    }

    offset.first   += block_size*work_per_thread;
    offset.second  += block_size*work_per_thread;
    processed_size += block_size*work_per_thread;
    remaining_size -= block_size*work_per_thread;

    if(threadIdx.x == block_size-1)
    {
      block_begin = thread_begin;
    }
    __syncthreads();
  } // end while
}


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


