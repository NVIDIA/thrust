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

#pragma once

#include <thrust/system/cuda/detail/bulk/detail/config.hpp>
#include <thrust/system/cuda/detail/bulk/algorithm/copy.hpp>
#include <thrust/system/cuda/detail/bulk/malloc.hpp>
#include <thrust/system/cuda/detail/bulk/uninitialized.hpp>
#include <thrust/system/cuda/detail/bulk/iterator/strided_iterator.hpp>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/detail/minmax.h>


BULK_NAMESPACE_PREFIX
namespace bulk
{


template<std::size_t bound,
         std::size_t grainsize,
         typename RandomAccessIterator,
         typename T,
         typename BinaryFunction>
__forceinline__ __device__
T reduce(const bulk::bounded<bound,bulk::agent<grainsize> > &exec,
         RandomAccessIterator first,
         RandomAccessIterator last,
         T init,
         BinaryFunction binary_op)
{
  typedef typename bulk::bounded<bound,bulk::agent<grainsize> >::size_type size_type;

  size_type n = last - first;

  for(size_type i = 0; i < exec.bound(); ++i)
  {
    if(i < n)
    {
      init = binary_op(init, first[i]);
    } // end if
  } // end for i

  return init;
} // end reduce()


namespace detail
{
namespace reduce_detail
{


template<typename ConcurrentGroup, typename RandomAccessIterator, typename Size, typename T, typename BinaryFunction>
__device__ T destructive_reduce_n(ConcurrentGroup &g, RandomAccessIterator first, Size n, T init, BinaryFunction binary_op)
{
  typedef int size_type;

  size_type tid = g.this_exec.index();

  Size m = n;

  while(m > 1)
  {
    Size half_m = m >> 1;

    if(tid < half_m)
    {
      T old_val = first[tid];

      first[tid] = binary_op(old_val, first[m - tid - 1]);
    } // end if

    g.wait();

    m -= half_m;
  } // end while

  g.wait();

  T result = init;
  if(n > 0)
  {
    result = binary_op(result,first[0]);
  } // end if

  g.wait();

  return result;
} // end destructive_reduce_n()


} // end reduce_detail
} // end detail


template<std::size_t groupsize, std::size_t grainsize, typename RandomAccessIterator, typename T, typename BinaryFunction>
__device__
T reduce(bulk::concurrent_group<bulk::agent<grainsize>,groupsize> &g,
         RandomAccessIterator first,
         RandomAccessIterator last,
         T init,
         BinaryFunction binary_op)
{
  typedef int size_type;

  const size_type elements_per_group = groupsize * grainsize;

  size_type tid = g.this_exec.index();

  T this_sum;

  bool this_sum_defined = false;

  size_type n = last - first;

  // XXX we use offset as the loop counter variable instead of first
  //     because elements_per_group can actually overflow some kinds of iterators
  //     with small difference_types
  for(size_type offset = 0; offset < n; first += elements_per_group, offset += elements_per_group)
  {
    size_type partition_size = thrust::min<size_type>(elements_per_group, last - first);

    typedef typename thrust::iterator_value<RandomAccessIterator>::type input_type;
    
    // load input into register
    input_type local_inputs[grainsize];

    // each agent strides through the input range
    // and copies into a local array
    strided_iterator<RandomAccessIterator,size_type> local_first = make_strided_iterator(first + tid, static_cast<size_type>(groupsize));

    // XXX if we could precompute local_size for the else branch,
    //     we could just call copy_n here
    //     we can't precompute it (without a divide afaik), so we compute local_size in the else branch
    size_type local_size = 0;
    if(partition_size < elements_per_group)
    {
//  XXX i guess nvcc miscompiles this loop for counting_iterators
//      size_type index = tid;
//      for(size_type i = 0; i < grainsize; ++i, ++local_first, index += groupsize)
//      {
//        if(index < partition_size)
//        {
//          local_inputs[i] = *local_first;
//          ++local_size;
//        } // end if
//      } // end for
//
      RandomAccessIterator iter = local_first.base();
      size_type index = tid;
      for(size_type i = 0; i < grainsize; ++i, index += groupsize, iter += groupsize)
      {
        if(index < partition_size)
        {
          local_inputs[i] = *iter;
          ++local_size;
        } // end if
      } // end for
    } // end if
    else
    {
      local_size = grainsize;
//  XXX nvcc 6.5 RC miscompiles this loop when RandomAccessIterator is a counting_iterator
//      bulk::copy_n(bulk::bound<grainsize>(g.this_exec),
//                   local_first,
//                   local_size,
//                   local_inputs);
      RandomAccessIterator iter = local_first.base();
      for(size_type i = 0; i < grainsize; ++i, iter += groupsize)
      {
        local_inputs[i] = *iter;
      } // end for
    } // end else

    // reduce local_inputs sequentially
    this_sum = this_sum_defined ?
      bulk::reduce(bulk::bound<grainsize>(g.this_exec), local_inputs, local_inputs + local_size, this_sum, binary_op) :
      bulk::reduce(bulk::bound<grainsize-1>(g.this_exec), local_inputs + 1, local_inputs + local_size, T(local_inputs[0]), binary_op);

    this_sum_defined = true;
  } // end for

#if __CUDA_ARCH__ >= 200
  T *buffer = reinterpret_cast<T*>(bulk::malloc(g, groupsize * sizeof(T)));
#else
  __shared__ bulk::uninitialized_array<T,groupsize> buffer_impl;
  T *buffer = buffer_impl.data();
#endif

  if(this_sum_defined)
  {
    buffer[tid] = this_sum;
  } // end if

  g.wait();

  // reduce across the group
  T result = bulk::detail::reduce_detail::destructive_reduce_n(g, buffer, thrust::min<size_type>(groupsize,n), init, binary_op);

#if __CUDA_ARCH__ >= 200
  bulk::free(g,buffer);
#endif

  return result;
} // end reduce


template<typename RandomAccessIterator, typename T, typename BinaryFunction>
__device__
T reduce(bulk::concurrent_group<> &g,
         RandomAccessIterator first,
         RandomAccessIterator last,
         T init,
         BinaryFunction binary_op)
{
  typedef int size_type;

  size_type tid = g.this_exec.index();

  T this_sum;

  bool this_sum_defined = false;

  typename thrust::iterator_difference<RandomAccessIterator>::type n = last - first;

  T *buffer = reinterpret_cast<T*>(bulk::malloc(g, g.size() * sizeof(T)));

  for(size_type i = tid; i < n; i += g.size())
  {
    typedef typename thrust::iterator_value<RandomAccessIterator>::type input_type;
    input_type x = first[i];
    this_sum = this_sum_defined ? binary_op(this_sum, x) : x;

    this_sum_defined = true;
  }

  if(this_sum_defined)
  {
    buffer[tid] = this_sum;
  } // end if

  g.wait();

  // reduce across the block
  T result = detail::reduce_detail::destructive_reduce_n(g, buffer, thrust::min<size_type>(g.size(),n), init, binary_op);

  bulk::free(g,buffer);

  return result;
} // end reduce


} // end bulk
BULK_NAMESPACE_SUFFIX

