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
#include <thrust/system/cuda/detail/bulk/algorithm/reduce.hpp>
#include <thrust/system/cuda/detail/bulk/execution_policy.hpp>
#include <thrust/system/cuda/detail/bulk/uninitialized.hpp>
#include <thrust/detail/type_traits/function_traits.h>

BULK_NAMESPACE_PREFIX
namespace bulk
{


template<std::size_t bound,
         std::size_t grainsize,
         typename RandomAccessIterator,
         typename T,
         typename BinaryFunction>
__forceinline__ __device__
T accumulate(const bounded<bound,bulk::agent<grainsize> > &exec,
             RandomAccessIterator first,
             RandomAccessIterator last,
             T init,
             BinaryFunction binary_op)
{
  typedef typename bounded<bound,bulk::agent<grainsize> >::size_type size_type;

  size_type n = last - first;

  for(size_type i = 0; i < exec.bound(); ++i)
  {
    if(i < n)
    {
      init = binary_op(init, first[i]);
    } // end if
  } // end for i

  return init;
} // end accumulate()


namespace detail
{
namespace accumulate_detail
{


// XXX this implementation is simply an inplace inclusive scan
//     we could potentially do better with an implementation which uses Sean's bitfield reverse trick
template<typename ConcurrentGroup, typename RandomAccessIterator, typename Size, typename T, typename BinaryFunction>
__device__ T destructive_accumulate_n(ConcurrentGroup &g, RandomAccessIterator first, Size n, T init, BinaryFunction binary_op)
{
  typedef typename ConcurrentGroup::size_type size_type;

  size_type tid = g.this_exec.index();

  T x = init;
  if(tid < n)
  {
    x = first[tid];
  }

  g.wait();

  for(size_type offset = 1; offset < g.size(); offset += offset)
  {
    if(tid >= offset && tid - offset < n)
    {
      x = binary_op(first[tid - offset], x);
    }

    g.wait();

    if(tid < n)
    {
      first[tid] = x;
    }

    g.wait();
  }

  return binary_op(init, first[n - 1]);
}


template<std::size_t groupsize, std::size_t grainsize, typename RandomAccessIterator, typename T>
struct buffer
{
  typedef typename thrust::iterator_value<RandomAccessIterator>::type value_type;

  union
  {
    uninitialized_array<value_type, groupsize * grainsize> inputs;
    uninitialized_array<T, groupsize>                      sums;
  }; // end union
}; // end buffer


template<std::size_t groupsize, std::size_t grainsize, typename RandomAccessIterator, typename T, typename BinaryFunction>
__device__
T accumulate(bulk::concurrent_group<bulk::agent<grainsize>,groupsize> &g,
             RandomAccessIterator first,
             RandomAccessIterator last,
             T init,
             BinaryFunction binary_op)
{
  typedef typename bulk::concurrent_group<bulk::agent<grainsize>,groupsize>::size_type size_type;

  const size_type elements_per_group = groupsize * grainsize;

  size_type tid = g.this_exec.index();

  T sum = init;

  typename thrust::iterator_difference<RandomAccessIterator>::type n = last - first;

  typedef detail::accumulate_detail::buffer<
    groupsize,
    grainsize,
    RandomAccessIterator,
    T
  > buffer_type;

#if __CUDA_ARCH__ >= 200
  buffer_type *buffer = reinterpret_cast<buffer_type*>(bulk::malloc(g, sizeof(buffer_type)));
#else
  __shared__ uninitialized<buffer_type> buffer_impl;
  buffer_type *buffer = &buffer_impl.get();
#endif
  
  for(; first < last; first += elements_per_group)
  {
    // XXX each iteration is essentially a bounded accumulate
    
    size_type partition_size = thrust::min<size_type>(elements_per_group, last - first);
    
    // copy partition into smem
    bulk::copy_n(g, first, partition_size, buffer->inputs.data());
    
    T this_sum;
    size_type local_offset = grainsize * g.this_exec.index();

    size_type local_size = thrust::max<size_type>(0,thrust::min<size_type>(grainsize, partition_size - grainsize * tid));

    if(local_size)
    {
      this_sum = buffer->inputs[local_offset];
      this_sum = bulk::accumulate(bound<grainsize-1>(g.this_exec),
                                  buffer->inputs.data() + local_offset + 1,
                                  buffer->inputs.data() + local_offset + local_size,
                                  this_sum,
                                  binary_op);
    } // end if

    g.wait();

    if(local_size)
    {
      buffer->sums[tid] = this_sum;
    } // end if

    g.wait();
    
    // sum over the group
    sum = accumulate_detail::destructive_accumulate_n(g, buffer->sums.data(), thrust::min<size_type>(groupsize,n), sum, binary_op);
  } // end for

#if __CUDA_ARCH__ >= 200
  bulk::free(g, buffer);
#endif

  return sum;
} // end accumulate
} // end accumulate_detail
} // end detail


template<std::size_t groupsize, std::size_t grainsize, typename RandomAccessIterator, typename T, typename BinaryFunction>
__device__
T accumulate(bulk::concurrent_group<bulk::agent<grainsize>, groupsize> &g,
             RandomAccessIterator first,
             RandomAccessIterator last,
             T init,
             BinaryFunction binary_op)
{
  // use reduce when the operator is commutative
  if(thrust::detail::is_commutative<BinaryFunction>::value)
  {
    init = bulk::reduce(g, first, last, init, binary_op);
  } // end if
  else
  {
    init = detail::accumulate_detail::accumulate(g, first, last, init, binary_op);
  } // end else

  return init;
} // end accumulate()


} // end bulk
BULK_NAMESPACE_SUFFIX

