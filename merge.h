#include <thrust/detail/config.h>
#include <thrust/pair.h>
#include <thrust/detail/minmax.h>
#include <thrust/detail/function.h>
#include <thrust/system/cuda/detail/detail/uninitialized.h>


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
  Size end = thrust::min<Size>(diag - lower_bound2, upper_bound1);

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


template<unsigned int block_size, unsigned int per_thread, typename RandomAccessIterator1, typename Size, typename RandomAccessIterator2, typename RandomAccessIterator3, typename Compare>
__global__
void merge_n(RandomAccessIterator1 first1,
             Size n1,
             RandomAccessIterator2 first2,
             Size n2,
             RandomAccessIterator3 result,
             Compare comp_)
{
  thrust::detail::device_function<Compare,bool> comp;
  typedef typename thrust::iterator_value<RandomAccessIterator1>::type value_type1;
  typedef typename thrust::iterator_value<RandomAccessIterator2>::type value_type2;

  Size result_size = n1 + n2;
  Size per_cta_size = (result_size-1)/gridDim.x+1;
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
                       Size(cta_offset+processed_size+threadIdx.x*per_thread),
                       block_begin.get().first, thrust::min<Size>(offset.first + block_size*per_thread, n1),
                       block_begin.get().second, thrust::min<Size>(offset.second + block_size*per_thread, n2),
                       comp);

    value_type1 x1 = first1[thread_begin.first];
    value_type2 x2 = first2[thread_begin.second];

    Size count = thrust::min<Size>(per_thread, result_size - thread_begin.first - thread_begin.second);
    for(Size i = 0; i < count; ++i)
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
         p = !comp(x2,x1);
       }

       result[cta_offset+processed_size+threadIdx.x*per_thread+i] = p ? x1 : x2;

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

    offset.first   += block_size*per_thread;
    offset.second  += block_size*per_thread;
    processed_size += block_size*per_thread;
    remaining_size -= block_size*per_thread;

    if(threadIdx.x == block_size-1)
    {
      block_begin = thread_begin;
    }
    __syncthreads();
  } // end while
}

