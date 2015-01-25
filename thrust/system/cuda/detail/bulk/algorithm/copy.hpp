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
#include <thrust/system/cuda/detail/bulk/execution_policy.hpp>
#include <thrust/system/cuda/detail/bulk/detail/is_contiguous_iterator.hpp>
#include <thrust/system/cuda/detail/bulk/detail/pointer_traits.hpp>
#include <thrust/detail/type_traits.h>


BULK_NAMESPACE_PREFIX
namespace bulk
{


template<std::size_t bound,
         std::size_t grainsize,
         typename RandomAccessIterator1,
         typename Size,
         typename RandomAccessIterator2>
__forceinline__ __device__
RandomAccessIterator2 copy_n(const bounded<bound,agent<grainsize> > &b,
                             RandomAccessIterator1 first,
                             Size n,
                             RandomAccessIterator2 result)
{
  typedef typename bounded<bound,agent<grainsize> >::size_type size_type;

  if(bound <= n)
  {
    for(size_type i = 0; i < b.bound(); ++i, ++result, ++first)
    {
      *result = *first;
    } // end for i
  } // end if
  else
  {
    for(size_type i = 0; i < b.bound(); ++i, ++first)
    {
      if(i < n)
      {
        *result = *first;
        ++result;
      } // end if
    } // end for i
  } // end else

  return result;
} // end copy_n()



namespace detail
{


template<typename ConcurrentGroup,
         typename RandomAccessIterator1,
         typename Size,
         typename RandomAccessIterator2>
__forceinline__ __device__
RandomAccessIterator2 simple_copy_n(ConcurrentGroup &g, RandomAccessIterator1 first, Size n, RandomAccessIterator2 result)
{
  for(Size i = g.this_exec.index();
      i < n;
      i += g.size())
  {
    result[i] = first[i];
  } // end for i

  g.wait();

  return result + n;
} // end simple_copy_n()


template<std::size_t size,
         std::size_t grainsize,
         typename RandomAccessIterator1,
         typename Size,
         typename RandomAccessIterator2>
__forceinline__ __device__
typename thrust::detail::enable_if<
  (size * grainsize > 0),
  RandomAccessIterator2
>::type
  simple_copy_n(bulk::concurrent_group<
                  agent<grainsize>,
                  size
                > &g,
                RandomAccessIterator1 first, Size n,
                RandomAccessIterator2 result)
{
  typedef bulk::concurrent_group<
    agent<grainsize>,
    size
  > group_type;

  RandomAccessIterator2 return_me = result + n;

  typedef typename group_type::size_type size_type;
  size_type chunk_size = size * grainsize;

  size_type tid = g.this_exec.index();

  // important special case which avoids the expensive for loop below
  if(chunk_size == n)
  {
    // offset iterators by tid before loop
    first += tid;
    result += tid;

    for(size_type i = 0; i < grainsize; ++i, first += size, result += size)
    {
      *result = *first;
    } // end for
  } // end if
  else
  {
    // XXX i have a feeling the indexing could be rewritten to require less arithmetic
    for(RandomAccessIterator1 last = first + n;
        first < last;
        first += chunk_size, result += chunk_size)
    {
      // avoid conditional accesses when possible
      if((last - first) >= chunk_size)
      {
        for(size_type i = 0; i < grainsize; ++i)
        {
          size_type idx = size * i + tid;
          result[idx] = first[idx];
        } // end for
      } // end if
      else
      {
        for(size_type i = 0; i < grainsize; ++i)
        {
          size_type idx = size * i + tid;
          if(idx < (last - first))
          {
            result[idx] = first[idx];
          } // end if
        } // end for
      } // end else
    } // end for
  } // end else

  g.wait();

  return return_me;
} // end simple_copy_n()


template<std::size_t size,
         std::size_t grainsize,
         typename RandomAccessIterator1,
         typename Size,
         typename RandomAccessIterator2>
__forceinline__ __device__
RandomAccessIterator2 copy_n(concurrent_group<
                               agent<grainsize>,
                               size
                             > &g,
                             RandomAccessIterator1 first,
                             Size n,
                             RandomAccessIterator2 result)
{
  return detail::simple_copy_n(g, first, n, result);
} // end copy_n()


} // end detail


template<std::size_t groupsize,
         typename Executor,
         typename RandomAccessIterator1,
         typename Size,
         typename RandomAccessIterator2>
__forceinline__ __device__
RandomAccessIterator2
  copy_n(bulk::concurrent_group<Executor,groupsize> &g, RandomAccessIterator1 first, Size n, RandomAccessIterator2 result)
{
  return detail::copy_n(g, first, n, result);
} // end copy_n()


template<std::size_t bound, std::size_t groupsize, std::size_t grainsize, typename RandomAccessIterator1, typename Size, typename RandomAccessIterator2>
__device__
typename thrust::detail::enable_if<
  (bound <= groupsize * grainsize),
  RandomAccessIterator2 
>::type
copy_n(bulk::bounded<
         bound,
         concurrent_group<
           agent<grainsize>,
           groupsize
         >
       > &g,
       RandomAccessIterator1 first,
       Size n,
       RandomAccessIterator2 result)
{
  typedef bounded<
    bound,
    concurrent_group<
      agent<grainsize>,
      groupsize
    >
  > group_type;

  typedef typename group_type::size_type size_type;

  size_type tid = g.this_exec.index();

  typedef typename thrust::iterator_value<RandomAccessIterator1>::type value_type;

  // XXX make this an uninitialized array
  value_type stage[grainsize];

  // avoid conditional accesses when possible
  if(groupsize * grainsize <= n)
  {
    for(size_type i = 0; i < grainsize; ++i)
    {
      size_type src_idx = g.size() * i + tid;
      stage[i] = first[src_idx];
    } // end for i

    for(size_type i = 0; i < grainsize; ++i)
    {
      size_type dst_idx = g.size() * i + tid;
      result[dst_idx] = stage[i];
    } // end for i
  } // end if
  else
  {
    for(size_type i = 0; i < grainsize; ++i)
    {
      size_type src_idx = g.size() * i + tid;
      if(src_idx < n)
      {
        stage[i] = first[src_idx];
      } // end if
    } // end for

    for(size_type i = 0; i < grainsize; ++i)
    {
      size_type dst_idx = g.size() * i + tid;
      if(dst_idx < n)
      {
        result[dst_idx] = stage[i];
      } // end if
    } // end for
  } // end else

  g.wait();

  return result + thrust::min<Size>(g.size() * grainsize, n);
} // end copy_n()


} // end bulk
BULK_NAMESPACE_SUFFIX

